"""Lectura, preprocesamiento y Dataset de Sen1Floods11 (v1.1) y de la región de prueba de Lima.

Estructura esperada (la misma del bucket público gs://sen1floods11/v1.1):
  DATA_ROOT/data/flood_events/HandLabeled/{S1Hand,LabelHand,S1OtsuLabelHand,S2Hand,JRCWaterHand}/
  DATA_ROOT/data/flood_events/WeaklyLabeled/{S1Weak,S1OtsuLabelWeak,S2IndexLabelWeak}/
  DATA_ROOT/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv
  DATA_ROOT/peru/{S1Hand,LabelHand}/ + DATA_ROOT/peru/flood_peru_data.csv   (lo genera peru_prep.py)

Codificación de etiquetas (verificada en los archivos): -1 = sin dato / no válido, 0 = no agua, 1 = agua.
Las etiquetas S1OtsuLabelWeak solo tienen {0,1}; S2IndexLabelWeak tiene {-1,0,1} (-1 = nube).
S1: 2 bandas float32 en dB, banda 0 = VV, banda 1 = VH. Puede traer NaN (bordes de escena).
"""
from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = -1
HAND = "data/flood_events/HandLabeled"
WEAK = "data/flood_events/WeaklyLabeled"
SPLITS = "splits/flood_handlabeled"

# Rango de recorte en dB. Es el mismo del código oficial (np.clip(x, -50, 1)); fuera de ese rango
# solo hay valores atípicos (esquinas brillantes o ruido térmico).
CLIP_DB = (-50.0, 1.0)


def read_tif(path: str) -> np.ndarray:
    """Devuelve (C, H, W). Usa tifffile (sin GDAL); si falla, intenta con rasterio."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no existe {path} (¿se descargó esa carpeta? ver src/download.py)")
    try:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim == 3 and a.shape[-1] < a.shape[0]:   # tifffile da (H, W, C) para multibanda "contig"
            a = np.moveaxis(a, -1, 0)
    except Exception:
        import rasterio
        with rasterio.open(path) as src:
            a = src.read()
    if a.ndim == 2:
        a = a[None]
    return a


def chip_id(fname: str) -> str:
    """'Ghana_103272_S1Hand.tif' -> 'Ghana_103272'."""
    return "_".join(os.path.basename(fname).split("_")[:2])


def event_of(cid: str) -> str:
    return cid.split("_")[0]


@dataclass
class Sample:
    img: str
    label: str
    source: str          # "hand", "weak_s1", "weak_s2", "peru"
    s2: str | None = None


def hand_split(root: str, split: str) -> list[Sample]:
    """split en {train, valid, test, bolivia}. Usa los CSV oficiales."""
    path = os.path.join(root, SPLITS, f"flood_{split}_data.csv")
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            cid = chip_id(row[0])
            out.append(Sample(os.path.join(root, HAND, "S1Hand", f"{cid}_S1Hand.tif"),
                              os.path.join(root, HAND, "LabelHand", f"{cid}_LabelHand.tif"), "hand",
                              os.path.join(root, HAND, "S2Hand", f"{cid}_S2Hand.tif")))
    return out


def weak_samples(root: str, kind: str, exclude_events: tuple[str, ...] = ("Bolivia",)) -> list[Sample]:
    """kind = 'weak_s1' (Otsu sobre VH de S1) o 'weak_s2' (índices NDVI/MNDWI de Sentinel-2).

    Por defecto se EXCLUYE Bolivia: si Bolivia es la región de respaldo para generalización, sus chips
    débiles (224) son vecinos de los 15 chips de prueba y filtrarían información de esa región.
    """
    sub = {"weak_s1": "S1OtsuLabelWeak", "weak_s2": "S2IndexLabelWeak"}[kind]
    out = []
    for lab in sorted(glob.glob(os.path.join(root, WEAK, sub, "*.tif"))):
        cid = chip_id(lab)
        if event_of(cid) in exclude_events:
            continue
        out.append(Sample(os.path.join(root, WEAK, "S1Weak", f"{cid}_S1Weak.tif"), lab, kind))
    return out


def peru_samples(root: str) -> list[Sample]:
    path = os.path.join(root, "peru", "flood_peru_data.csv")
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if row:
                out.append(Sample(os.path.join(root, "peru", "S1Hand", row[0]),
                                  os.path.join(root, "peru", "LabelHand", row[1]), "peru"))
    return out


# ----------------------------------------------------------------------------- preprocesamiento
def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Filtro de Lee (1980) sobre intensidad LINEAL, banda por banda; devuelve dB.

    El speckle es multiplicativo en intensidad lineal, por eso se filtra antes de pasar a dB.
    """
    from scipy.ndimage import uniform_filter
    out = np.empty_like(img)
    for b in range(img.shape[0]):
        lin = 10 ** (img[b] / 10.0)
        mean = uniform_filter(lin, size)
        sq_mean = uniform_filter(lin ** 2, size)
        var = np.maximum(sq_mean - mean ** 2, 0)
        noise_var = np.mean(var)                      # estimación global simple del ruido
        w = var / (var + noise_var + 1e-12)
        out[b] = 10 * np.log10(np.maximum(mean + w * (lin - mean), 1e-6))
    return out


def preprocess_s1(x_db: np.ndarray, mean: np.ndarray, std: np.ndarray, speckle: str = "none",
                  add_ratio: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """dB crudos (2,H,W) -> (tensor normalizado, máscara de píxeles válidos).

    1. Los NaN se marcan como no válidos (el código oficial los convertía en 0 dB, que para un radar es
       un retorno MUY brillante: introduce un falso "no agua" en los bordes de escena).
    2. Recorte a CLIP_DB y, opcional, filtro de Lee.
    3. Opcional: canal VV-VH (cociente en dB), útil para separar agua (cociente bajo) de suelo liso.
    4. Estandarización por banda con media/desv. calculadas SOLO en train (no se mira test).
    """
    valid = np.isfinite(x_db).all(axis=0)
    x = np.where(np.isfinite(x_db), x_db, np.nan)
    fill = mean[:2, None, None] if mean.shape[0] >= 2 else 0
    x = np.where(np.isnan(x), fill, x)
    x = np.clip(x, *CLIP_DB).astype(np.float32)
    if speckle == "lee":
        x = lee_filter(x)
    if add_ratio:
        x = np.concatenate([x, x[0:1] - x[1:2]], axis=0)
    x = (x - mean[: x.shape[0], None, None]) / std[: x.shape[0], None, None]
    return x.astype(np.float32), valid


def preprocess_s2(s2: np.ndarray, bands=(1, 2, 3, 7, 10, 11)) -> np.ndarray:
    """Bono: bandas ópticas (B2,B3,B4,B8,B11,B12 por defecto), reflectancia TOA escalada x10000."""
    return np.clip(s2[list(bands)].astype(np.float32) / 10000.0, 0, 1) * 4 - 1   # aprox. en [-1, 3]


def compute_stats(samples: list[Sample], add_ratio: bool = False, max_chips: int | None = None):
    """Media y desviación por canal en dB (sobre píxeles válidos) del conjunto de ENTRENAMIENTO."""
    s, s2, n = None, None, 0
    for smp in samples[:max_chips]:
        x = read_tif(smp.img).astype(np.float64)
        ok = np.isfinite(x).all(axis=0)
        x = np.clip(np.where(np.isfinite(x), x, 0), *CLIP_DB)
        if add_ratio:
            x = np.concatenate([x, x[0:1] - x[1:2]], axis=0)
        v = x[:, ok]
        s = v.sum(1) if s is None else s + v.sum(1)
        s2 = (v ** 2).sum(1) if s2 is None else s2 + (v ** 2).sum(1)
        n += v.shape[1]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------------- Dataset
class FloodDataset(Dataset):
    """Devuelve (x, y, w): x (C,H,W) float; y (H,W) float en {0,1} o IGNORE; w peso escalar del chip.

    El peso permite combinar fuentes: los chips débiles pueden pesar menos que los dorados (lambda).
    """

    def __init__(self, samples, mean, std, crop: int | None = 256, augment: bool = False,
                 speckle: str = "none", add_ratio: bool = False, use_s2: bool = False,
                 weights: dict[str, float] | None = None, cache: bool = True):
        self.samples, self.mean, self.std = samples, mean, std
        self.crop, self.augment, self.speckle, self.add_ratio, self.use_s2 = crop, augment, speckle, add_ratio, use_s2
        self.weights = weights or {}
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.samples)

    def _load(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        smp = self.samples[i]
        x, valid = preprocess_s1(read_tif(smp.img).astype(np.float32), self.mean, self.std,
                                 self.speckle, self.add_ratio)
        y = read_tif(smp.label)[0].astype(np.int16)
        y = np.where(valid, y, IGNORE)          # sin dato SAR => no se evalúa ni se entrena ese píxel
        if self.use_s2:
            s2 = preprocess_s2(read_tif(smp.s2))
            x = np.concatenate([x, s2], axis=0)
        out = (x, y)
        if self.cache is not None:
            self.cache[i] = out
        return out

    def __getitem__(self, i):
        x, y = self._load(i)
        if self.crop and self.crop < x.shape[-1]:
            r = random.randint(0, x.shape[-2] - self.crop)
            c = random.randint(0, x.shape[-1] - self.crop)
            x, y = x[:, r:r + self.crop, c:c + self.crop], y[r:r + self.crop, c:c + self.crop]
        if self.augment:
            # Solo volteos: rotar 90° cambiaría la dirección de iluminación del radar (la sombra y el
            # layover siempre caen hacia el mismo lado de la escena según la órbita).
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        w = self.weights.get(self.samples[i].source, 1.0)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)).float(),
                torch.tensor(w, dtype=torch.float32))
