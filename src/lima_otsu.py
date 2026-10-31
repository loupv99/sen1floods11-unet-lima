"""Etiqueta Otsu-VH para los chips de Lima, como S1OtsuLabelHand de Sen1Floods11 (un umbral por evento sobre VH).

Uso:  python -m src.lima_otsu --root data/sen1floods11/v1.1     (escribe <root>/peru/S1OtsuLabelHand/*.tif)
"""
import argparse
import glob
import os

import numpy as np
import rasterio


def otsu(v, bins=512):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.root, "peru", "S1Hand", "*_S1Hand.tif")))
vh = []
for f in files:
    with rasterio.open(f) as s:
        x = s.read(2)
    vh.append(x[np.isfinite(x)])
thr = otsu(np.clip(np.concatenate(vh), -50, 1))
print(f"umbral Otsu VH de la escena de Lima: {thr:.2f} dB")
out = os.path.join(a.root, "peru", "S1OtsuLabelHand"); os.makedirs(out, exist_ok=True)
for f in files:
    with rasterio.open(f) as s:
        x = s.read(2); prof = s.profile
    y = np.where(np.isfinite(x), (x < thr).astype(np.int16), -1).astype(np.int16)
    prof.update(count=1, dtype="int16", nodata=None)
    with rasterio.open(os.path.join(out, os.path.basename(f).replace("S1Hand", "S1OtsuLabelHand")), "w", **prof) as d:
        d.write(y[None])
print(f"{len(files)} etiquetas Otsu escritas")
