"""Evaluación de un checkpoint (o de la línea base Otsu) sobre un split.

  python -m src.evaluate --root R --ckpt runs/A/best.pt --split test     # en distribución (entrega 3)
  python -m src.evaluate --root R --ckpt runs/A/best.pt --split bolivia  # región de respaldo (entrega 4)
  python -m src.evaluate --root R --ckpt runs/A/best.pt --split peru     # Lima 2017 (entrega 4)
  python -m src.evaluate --root R --baseline otsu --split test            # línea base clásica

Umbral: por defecto 0.5. Con --tune-threshold se elige el umbral que maximiza el IoU en VALID y luego
se aplica sin cambios al split pedido (nunca se elige mirando test).
Escribe en runs/<run>/eval_<split>/: metrics.json, per_chip.csv, confusion.txt y probs.npz
(probabilidades en uint8, para el análisis de fallos).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from .data import FloodDataset, chip_id, event_of, hand_split, peru_samples, read_tif
from .metrics import MetricAccumulator, confusion_matrix_text
from .unet import UNet


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    m = UNet(in_ch=ck["in_ch"], base=a["base"], depth=a["depth"], skip_mode=a["skip_mode"], norm=a["norm"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, ds, device):
    probs = []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        p = torch.sigmoid(model(x[None].to(device)))[0, 0].float().cpu().numpy()
        probs.append((p, y.numpy().astype(np.int16)))
    return probs


def score(probs, samples, thr):
    acc = MetricAccumulator()
    for (p, y), s in zip(probs, samples):
        cid = chip_id(s.img)
        acc.update((p >= thr).astype(np.uint8), y, cid, event_of(cid))
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "valid", "test", "bolivia", "peru"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--baseline", default="", choices=["", "otsu"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune-threshold", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = peru_samples(args.root) if args.split == "peru" else hand_split(args.root, args.split)

    if args.baseline == "otsu":
        # Línea base: la etiqueta "S1OtsuLabelHand" que trae el dataset (Otsu sobre VH, Bonafilia et al. 2020)
        acc = MetricAccumulator()
        for s in samples:
            cid = chip_id(s.img)
            y = read_tif(s.label)[0].astype(np.int16)
            x = read_tif(s.img)
            y = np.where(np.isfinite(x).all(0), y, -1)
            o = read_tif(s.label.replace("LabelHand", "S1OtsuLabelHand"))[0]
            acc.update((o == 1).astype(np.uint8), y, cid, event_of(cid))
        out_dir = args.out or os.path.join("runs", "baseline_otsu", f"eval_{args.split}")
        thr_used = None
    else:
        model, ck = load_model(args.ckpt, device)
        a = ck["args"]
        mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
        mk = lambda smp: FloodDataset(smp, mean, std, crop=None, augment=False, speckle=a["speckle"],
                                      add_ratio=a["add_ratio"], use_s2=a["use_s2"], cache=False)
        thr_used = args.threshold
        if args.tune_threshold:
            val_s = hand_split(args.root, "valid")
            vp = predict(model, mk(val_s), device)
            grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
            ious = [score(vp, val_s, t).summary()["pooled"]["IoU"] for t in grid]
            thr_used = float(grid[int(np.nanargmax(ious))])
            print("umbral elegido en VALID:", thr_used, " IoU val:", max(ious))
        probs = predict(model, mk(samples), device)
        acc = score(probs, samples, thr_used)
        out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "probs.npz"),
                            ids=np.array([chip_id(s.img) for s in samples]),
                            probs=np.stack([(p * 255).astype(np.uint8) for p, _ in probs]),
                            labels=np.stack([y for _, y in probs]))
    os.makedirs(out_dir, exist_ok=True)
    summ = acc.summary()
    summ["threshold"] = thr_used
    json.dump(summ, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "per_chip.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "evento", "TP", "FP", "FN", "TN", "IoU"])
        for cid, ev, tp, fp, fn, tn in acc.per_chip:
            u = tp + fp + fn
            w.writerow([cid, ev, tp, fp, fn, tn, f"{tp / u:.4f}" if u else ""])
    cm = confusion_matrix_text(acc.total)
    open(os.path.join(out_dir, "confusion.txt"), "w", encoding="utf-8").write(cm + "\n")
    p = summ["pooled"]
    print(f"[{args.split}] IoU {p['IoU']:.4f}  F1 {p['F1']:.4f}  P {p['precision']:.4f}  R {p['recall']:.4f}  "
          f"mIoU/chip(sin vacíos) {summ['mean_IoU_per_chip_excluding_empty']:.4f}  umbral {thr_used}")
    print(cm)


if __name__ == "__main__":
    main()
