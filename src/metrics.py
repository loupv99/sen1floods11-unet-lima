"""Métricas de la clase "agua", ignorando píxeles con etiqueta -1.

Se acumula la matriz de confusión en píxeles y se reportan DOS agregaciones, porque dan números
muy distintos y el informe debe decir cuál usa:
  - "pooled" (micro): se suman TP/FP/FN de todos los chips y luego se calcula IoU. Pesa más a los
    chips con mucha agua.
  - "mean per chip" (la de los autores de Sen1Floods11): IoU de cada chip y luego el promedio.
    Problema: un chip sin agua real ni predicha da 0/0. El código oficial le suma 1e-7 arriba y abajo,
    así que vale 1. Aquí se reportan las dos variantes: excluir esos chips y contarlos como 1.

Omisión  = FN / (TP + FN)  = 1 - recall      (agua real que el modelo no marcó)
Comisión = FP / (TP + FP)  = 1 - precisión   (agua marcada que no lo era)
(El paper de Sen1Floods11 define la comisión solo como "tasa de detecciones falsas positivas"; si se
quiere comparar con sus tablas, POR CONFIRMAR cuál denominador usaron. Se reporta también FPR.)
"""
from __future__ import annotations

import numpy as np

IGNORE = -1


def confusion(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """pred, y: arrays con 0/1 (y puede tener -1). Devuelve [TP, FP, FN, TN] en int64."""
    v = y != IGNORE
    p = (pred == 1) & v
    g = (y == 1) & v
    tp = np.sum(p & g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)
    tn = np.sum(v) - tp - fp - fn
    return np.array([tp, fp, fn, tn], dtype=np.int64)


def from_confusion(c: np.ndarray) -> dict:
    tp, fp, fn, tn = [float(v) for v in c]
    div = lambda a, b: a / b if b > 0 else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "IoU": div(tp, tp + fp + fn),
        "F1": div(2 * tp, 2 * tp + fp + fn),
        "precision": prec,
        "recall": rec,
        "omission": div(fn, tp + fn),
        "commission": div(fp, tp + fp),
        "FPR": div(fp, fp + tn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


class MetricAccumulator:
    def __init__(self):
        self.total = np.zeros(4, dtype=np.int64)
        self.per_chip = []      # (chip_id, evento, tp, fp, fn, tn)

    def update(self, pred, y, cid: str = "", event: str = ""):
        c = confusion(pred, y)
        self.total += c
        self.per_chip.append((cid, event, *c.tolist()))
        return c

    def summary(self) -> dict:
        out = {"pooled": from_confusion(self.total)}
        ious_skip, ious_one = [], []
        for _, _, tp, fp, fn, _ in self.per_chip:
            u = tp + fp + fn
            if u > 0:
                ious_skip.append(tp / u)
                ious_one.append(tp / u)
            else:
                ious_one.append(1.0)
        out["mean_IoU_per_chip_excluding_empty"] = float(np.mean(ious_skip)) if ious_skip else float("nan")
        out["mean_IoU_per_chip_empty_as_1"] = float(np.mean(ious_one)) if ious_one else float("nan")
        out["n_chips"] = len(self.per_chip)
        out["n_chips_empty"] = len(ious_one) - len(ious_skip)
        by_ev = {}
        for _, ev, *c in self.per_chip:
            by_ev.setdefault(ev, np.zeros(4, dtype=np.int64))
            by_ev[ev] += np.array(c, dtype=np.int64)
        out["per_event"] = {ev: from_confusion(c) for ev, c in sorted(by_ev.items())}
        return out


def confusion_matrix_text(c: np.ndarray) -> str:
    tp, fp, fn, tn = c
    tot = c.sum()
    return ("                  pred no-agua     pred agua\n"
            f"real no-agua  {tn:14,d} {fp:14,d}   ({100 * tn / tot:5.1f} % | {100 * fp / tot:5.1f} %)\n"
            f"real agua     {fn:14,d} {tp:14,d}   ({100 * fn / tot:5.1f} % | {100 * tp / tot:5.1f} %)")
