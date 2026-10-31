"""Entrenamiento de la U-Net en Sen1Floods11.

Ejemplos (ver run_all.sh para la lista completa de la ablación):
  python -m src.train --root data/sen1floods11/v1.1 --run A_hand_combo --train-data hand --loss combo
  python -m src.train --root ... --run C_hand+s2_combo --train-data hand+weak_s2 --weak-weight 0.5
  python -m src.train --root ... --run D_s2pre --train-data weak_s2 --epochs 15
  python -m src.train --root ... --run D_s2pre_ft --train-data hand --init runs/D_s2pre/best.pt

Decisiones fijas (y por qué):
  - selección del mejor epoch con IoU (pooled) en el split VALID oficial: test y Bolivia no se miran;
  - semilla global + generador propio del DataLoader: dos corridas con la misma semilla ven los
    mismos recortes en el mismo orden, aunque el modelo consuma números aleatorios distintos;
  - el tiempo mide solo el entrenamiento y sincroniza la GPU antes de leer el reloj.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FloodDataset, compute_stats, hand_split, weak_samples
from .losses import make_loss
from .metrics import MetricAccumulator
from .unet import UNet, count_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_samples(args):
    samples = []
    parts = args.train_data.split("+")
    if "hand" in parts:
        samples += hand_split(args.root, "train") * args.hand_oversample
    for k in ("weak_s1", "weak_s2"):
        if k in parts:
            ws = weak_samples(args.root, k, exclude_events=tuple(args.exclude_weak_events.split(",")) if args.exclude_weak_events else ())
            if args.max_weak:
                random.Random(args.seed).shuffle(ws)
                ws = ws[: args.max_weak]
            samples += ws
    return samples


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    acc = MetricAccumulator()
    for x, y, _ in loader:
        prob = torch.sigmoid(model(x.to(device)))[:, 0].float().cpu().numpy()
        for p, t in zip(prob, y.numpy()):
            acc.update((p >= threshold).astype(np.uint8), t.astype(np.int16))
    return acc.summary()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="carpeta v1.1 de Sen1Floods11")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--train-data", default="hand", help="hand | hand+weak_s1 | hand+weak_s2 | weak_s2 | weak_s1")
    ap.add_argument("--weak-weight", type=float, default=0.5, help="peso por píxel de los chips débiles (lambda)")
    ap.add_argument("--hand-oversample", type=int, default=1, help="repetir los chips dorados k veces por epoch")
    ap.add_argument("--exclude-weak-events", default="Bolivia")
    ap.add_argument("--max-weak", type=int, default=0, help="0 = todos")
    ap.add_argument("--init", default="", help="checkpoint para fine-tuning (estrategia preentrenar-débil -> ajustar-dorado)")
    ap.add_argument("--loss", default="combo", choices=["bce", "dice", "combo", "focal"])
    ap.add_argument("--pos-weight", type=float, default=0.0, help="0 = automático (no-agua/agua en train dorado)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--skip-mode", default="concat", choices=["concat", "none", "attention"])
    ap.add_argument("--norm", default="bn", choices=["bn", "gn", "none"])
    ap.add_argument("--speckle", default="none", choices=["none", "lee"])
    ap.add_argument("--add-ratio", action="store_true")
    ap.add_argument("--use-s2", action="store_true", help="bono: fusión SAR + óptico (solo chips dorados)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-chips", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--max-val-chips", type=int, default=0, help="solo para pruebas rápidas")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.out, args.run)
    os.makedirs(out_dir, exist_ok=True)

    hand_train = hand_split(args.root, "train")
    # Estadísticas SIEMPRE del train dorado (así todos los modelos ven la misma normalización).
    mean, std = compute_stats(hand_train, add_ratio=args.add_ratio)
    train_s = build_train_samples(args)
    val_s = hand_split(args.root, "valid")
    if not train_s:
        raise SystemExit(f"0 chips de entrenamiento para --train-data {args.train_data}: ¿falta descargar "
                         "WeaklyLabeled o la ruta --root está mal?")
    if args.max_train_chips:
        train_s = train_s[: args.max_train_chips]
    if args.max_val_chips:
        val_s = val_s[: args.max_val_chips]

    if args.pos_weight <= 0:  # razón no-agua / agua en el train dorado (medida, no supuesta)
        from .data import read_tif
        pos = neg = 0
        for s in hand_train:
            y = read_tif(s.label)[0]
            pos += int((y == 1).sum()); neg += int((y == 0).sum())
        args.pos_weight = neg / max(pos, 1)

    weights = {"hand": 1.0, "weak_s1": args.weak_weight, "weak_s2": args.weak_weight}
    cache_train = len(train_s) <= 800
    ds_tr = FloodDataset(train_s, mean, std, crop=args.crop, augment=True, speckle=args.speckle,
                         add_ratio=args.add_ratio, use_s2=args.use_s2, weights=weights, cache=cache_train)
    ds_va = FloodDataset(val_s, mean, std, crop=None, augment=False, speckle=args.speckle,
                         add_ratio=args.add_ratio, use_s2=args.use_s2)
    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, generator=g, drop_last=True,
                       num_workers=args.workers if not cache_train else 0, pin_memory=device == "cuda")
    dl_va = DataLoader(ds_va, batch_size=4, shuffle=False, num_workers=0)

    in_ch = 2 + int(args.add_ratio) + (6 if args.use_s2 else 0)
    model = UNet(in_ch=in_ch, base=args.base, depth=args.depth, skip_mode=args.skip_mode, norm=args.norm).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["model"])
        mean, std = np.array(ck["mean"], dtype=np.float32), np.array(ck["std"], dtype=np.float32)
        ds_tr.mean, ds_tr.std, ds_va.mean, ds_va.std = mean, std, mean, std
    loss_fn = make_loss(args.loss, args.pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl_tr)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    print(f"[{args.run}] train={len(train_s)} chips  val={len(val_s)}  params={count_params(model):,}  "
          f"pos_weight={args.pos_weight:.2f}  device={device}")
    hist, best, t_train = [], -1.0, 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for x, y, w in dl_tr:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item(); nb += 1
        if device == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        val = evaluate(model, dl_va, device)
        iou = val["pooled"]["IoU"]
        hist.append({"epoch": ep, "train_loss": run_loss / max(nb, 1), "val_IoU": iou,
                     "val_mIoU_chip": val["mean_IoU_per_chip_excluding_empty"], "t_train_s": t_train})
        print(f"ep {ep:3d}  loss {run_loss / max(nb, 1):.4f}  val IoU {iou:.4f}  "
              f"val mIoU/chip {val['mean_IoU_per_chip_excluding_empty']:.4f}")
        if iou > best:
            best = iou
            torch.save({"model": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                        "args": vars(args), "in_ch": in_ch, "epoch": ep, "val_IoU": iou}, os.path.join(out_dir, "best.pt"))
    json.dump({"args": vars(args), "history": hist, "best_val_IoU": best, "train_time_s": t_train},
              open(os.path.join(out_dir, "train_log.json"), "w"), indent=2)
    print(f"[{args.run}] mejor IoU val = {best:.4f}  tiempo de entrenamiento = {t_train / 60:.1f} min")


if __name__ == "__main__":
    main()
