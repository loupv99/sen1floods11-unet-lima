"""Entrega 5: análisis cualitativo de fallos característicos del SAR.

Toma probs.npz de evaluate.py y, para cada chip, cuenta los errores y les asigna una ETIQUETA
HEURÍSTICA (una pista, no un diagnóstico; el grupo confirma cada caso mirando la figura):

  FN_brillante   agua real no detectada con VV > -12 dB. El agua abierta es especular (VV típico
                 < -15 dB); si el agua "brilla" es candidata a DOBLE REBOTE (agua + pared en zona
                 urbana, o agua + tallos en vegetación inundada).
  FP_oscuro      "agua" predicha donde no hay, con VH < -22 dB: superficie lisa y oscura que no es
                 agua (arena seca, asfalto, pista de aterrizaje, suelo muy húmedo) o SOMBRA DE RADAR
                 (pendiente que da la espalda al sensor). Para separar sombra de lo demás hace falta
                 un DEM; en la figura se ve como una franja oscura pegada a una ladera.
  FP_aislado     falsos positivos en manchas de < 20 px: ruido speckle que la red no suavizó.
  FN_borde       FN concentrados en el contorno del agua real: borde mezclado (píxel mitad agua).

Genera figuras (VV, VH, RGB de Sentinel-2 si está, etiqueta, predicción, mapa de errores) para los
chips con peor IoU y para los que más pesan en cada etiqueta.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from .data import HAND, read_tif


def tag_chip(vv, vh, y, pred):
    from scipy import ndimage
    v = y != -1
    fn = (pred == 0) & (y == 1) & v
    fp = (pred == 1) & (y == 0) & v
    tags = {
        "FN_brillante": int(np.sum(fn & (vv > -12))),
        "FP_oscuro": int(np.sum(fp & (vh < -22))),
    }
    lab, n = ndimage.label(fp)
    sizes = ndimage.sum(fp, lab, range(1, n + 1)) if n else np.array([])
    tags["FP_aislado"] = int(sizes[sizes < 20].sum()) if n else 0
    water = (y == 1)
    edge = water ^ ndimage.binary_erosion(water, iterations=2)
    tags["FN_borde"] = int(np.sum(fn & edge))
    tags["FN"], tags["FP"] = int(fn.sum()), int(fp.sum())
    return tags


def figure(path, vv, vh, rgb, y, pred, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = np.zeros(y.shape + (3,), dtype=np.float32)
    err[(pred == 1) & (y == 1)] = (1, 1, 1)          # acierto agua: blanco
    err[(pred == 1) & (y == 0)] = (1, 0, 0)          # FP: rojo
    err[(pred == 0) & (y == 1)] = (0, 0.4, 1)        # FN: azul
    err[y == -1] = (0.4, 0.4, 0.4)                   # sin dato: gris
    panels = [("VV (dB)", vv, dict(cmap="gray", vmin=-25, vmax=0)),
              ("VH (dB)", vh, dict(cmap="gray", vmin=-30, vmax=-5))]
    if rgb is not None:
        panels.append(("Sentinel-2 RGB", rgb, {}))
    panels += [("Etiqueta dorada", np.where(y == -1, np.nan, y), dict(cmap="Blues", vmin=0, vmax=1)),
               ("Predicción", pred, dict(cmap="Blues", vmin=0, vmax=1)),
               ("Errores (rojo FP, azul FN)", err, {})]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (t, im, kw) in zip(axs, panels):
        ax.imshow(im, **kw); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval-dir", required=True, help="carpeta con probs.npz (salida de evaluate.py)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-fig", type=int, default=4, help="figuras por etiqueta")
    args = ap.parse_args(argv)

    d = np.load(os.path.join(args.eval_dir, "probs.npz"))
    ids, probs, labels = d["ids"], d["probs"], d["labels"]
    rows = []
    cache = {}
    for cid, pr, y in zip(ids, probs, labels):
        s1p = os.path.join(args.root, HAND, "S1Hand", f"{cid}_S1Hand.tif")
        if not os.path.exists(s1p):
            s1p = os.path.join(args.root, "peru", "S1Hand", f"{cid}_S1Hand.tif")
        x = read_tif(s1p).astype(np.float32)
        pred = (pr.astype(np.float32) / 255 >= args.threshold).astype(np.uint8)
        t = tag_chip(x[0], x[1], y, pred)
        tp = int(np.sum((pred == 1) & (y == 1)))
        u = tp + t["FP"] + t["FN"]
        rows.append({"chip": str(cid), "IoU": tp / u if u else float("nan"), **t})
        cache[str(cid)] = (x, y, pred)

    out = os.path.join(args.eval_dir, "fallos")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fallos_por_chip.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    chosen = set()
    for key in ["FN_brillante", "FP_oscuro", "FP_aislado", "FN_borde"]:
        for r in sorted(rows, key=lambda r: -r[key])[: args.n_fig]:
            if r[key] > 0:
                chosen.add((key, r["chip"]))
    for r in sorted([r for r in rows if r["IoU"] == r["IoU"]], key=lambda r: r["IoU"])[: args.n_fig]:
        chosen.add(("peor_IoU", r["chip"]))
    for key, cid in sorted(chosen):
        x, y, pred = cache[cid]
        rgb = None
        s2p = os.path.join(args.root, HAND, "S2Hand", f"{cid}_S2Hand.tif")
        if os.path.exists(s2p):
            s2 = read_tif(s2p).astype(np.float32)
            rgb = np.clip(np.stack([s2[3], s2[2], s2[1]], -1) / 3000.0, 0, 1)   # B4, B3, B2
        r = next(r for r in rows if r["chip"] == cid)
        figure(os.path.join(out, f"{key}__{cid}.png"), x[0], x[1], rgb, y, pred,
               f"{cid} | {key} | IoU={r['IoU']:.2f} FP={r['FP']} FN={r['FN']}")
    print(f"{len(rows)} chips analizados; {len(chosen)} figuras en {out}")


if __name__ == "__main__":
    main()
