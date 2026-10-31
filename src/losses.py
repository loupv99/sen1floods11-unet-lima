"""Pérdidas para segmentación binaria con píxeles ignorados (y = -1) y peso por chip.

Desbalance medido en Sen1Floods11 (train dorado): ~9.5 % de los píxeles válidos son agua.
  - BCE: promedia sobre píxeles => la clase mayoritaria domina; se compensa con pos_weight.
  - Dice: mide solapamiento de la clase agua => insensible al número de píxeles "no agua".
  - Combinada (BCE + Dice): BCE da gradientes estables píxel a píxel; Dice optimiza algo cercano al IoU.
  - Focal (Lin et al., 2017): baja el peso de los píxeles fáciles; opción extra.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -1


def _flat(logits, y, w_chip):
    # logits (B,1,H,W) | y (B,H,W) | w_chip (B,)
    logits = logits[:, 0]
    valid = (y != IGNORE).float()
    target = torch.where(y == IGNORE, torch.zeros_like(y), y)
    w_pix = valid * w_chip[:, None, None]
    return logits, target, valid, w_pix


def bce_loss(logits, y, w_chip, pos_weight: float = 1.0):
    logits, target, valid, w_pix = _flat(logits, y, w_chip)
    pw = torch.tensor(pos_weight, device=logits.device)
    per_pix = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return (per_pix * w_pix).sum() / w_pix.sum().clamp_min(1.0)


def dice_loss(logits, y, w_chip, eps: float = 1.0):
    """Dice suave calculado sobre TODO el batch (no por chip): un chip sin agua no da 0/0."""
    logits, target, valid, w_pix = _flat(logits, y, w_chip)
    p = torch.sigmoid(logits)
    inter = (p * target * w_pix).sum()
    denom = (p * w_pix).sum() + (target * w_pix).sum()
    return 1 - (2 * inter + eps) / (denom + eps)


def focal_loss(logits, y, w_chip, alpha: float = 0.75, gamma: float = 2.0):
    logits, target, valid, w_pix = _flat(logits, y, w_chip)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    per_pix = a_t * (1 - p_t) ** gamma * bce
    return (per_pix * w_pix).sum() / w_pix.sum().clamp_min(1.0)


def make_loss(name: str, pos_weight: float = 1.0):
    if name == "bce":
        return lambda lg, y, w: bce_loss(lg, y, w, pos_weight)
    if name == "dice":
        return dice_loss
    if name == "combo":
        return lambda lg, y, w: bce_loss(lg, y, w, pos_weight) + dice_loss(lg, y, w)
    if name == "focal":
        return focal_loss
    raise ValueError(name)
