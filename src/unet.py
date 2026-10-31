"""U-Net implementada desde cero en PyTorch (sin bibliotecas de segmentación).

Entrega 1 del examen: recibe al menos las dos bandas SAR (VV, VH) y devuelve un logit por píxel
para la clase "agua". Cambios respecto del paper original (Ronneberger et al., 2015), cada uno con
su motivo:
  - padding=1 ("same"): la salida tiene el mismo tamaño que la entrada (512x512 -> 512x512), así no
    hay que recortar los mapas del codificador antes de concatenar (notas de clase §2.2);
  - BatchNorm (o GroupNorm si el batch es pequeño) después de cada conv: estabiliza el entrenamiento
    con pocos datos. El paper original no la usaba;
  - una sola salida + sigmoide (binario agua/no agua) en lugar de K=2 con softmax.

Las opciones existen para el estudio de ablación (entrega 2b):
  depth      número de niveles de downsampling (3 o 4)
  skip_mode  "concat" (U-Net), "none" (codificador-decodificador sin saltos) o "attention"
             (compuertas de atención de Oktay et al., 2018, implementadas aquí)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind: str, ch: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm desconocida: {kind}")


class DoubleConv(nn.Module):
    """(conv3x3 -> norm -> ReLU) x 2. Es el bloque básico de cada nivel."""

    def __init__(self, c_in: int, c_out: int, norm: str = "bn"):
        super().__init__()
        bias = norm == "none"  # con norm, el sesgo de la conv es redundante (la norm lo absorbe)
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=bias), norm_layer(norm, c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=bias), norm_layer(norm, c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Compuerta de atención aditiva (Oktay et al., 2018).

    g = señal del decodificador (más profunda, más semántica); x = mapa del codificador (salto).
    alpha = sigmoid(psi(ReLU(W_g g + W_x x)))  en [0,1] por píxel; la salida es x * alpha.
    Así el decodificador aprende a "apagar" zonas del salto que no aportan (p. ej. ruido speckle).
    """

    def __init__(self, c_g: int, c_x: int, c_int: int):
        super().__init__()
        self.w_g = nn.Conv2d(c_g, c_int, 1, bias=True)
        self.w_x = nn.Conv2d(c_x, c_int, 1, bias=False)
        self.psi = nn.Conv2d(c_int, 1, 1, bias=True)

    def forward(self, g, x):
        a = F.relu(self.w_g(g) + self.w_x(x))
        alpha = torch.sigmoid(self.psi(a))
        return x * alpha


class Up(nn.Module):
    """Sube resolución x2 con convolución transpuesta y fusiona con el salto del codificador."""

    def __init__(self, c_in: int, c_skip: int, c_out: int, skip_mode: str, norm: str):
        super().__init__()
        self.skip_mode = skip_mode
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        c_up = c_in // 2
        if skip_mode == "attention":
            self.gate = AttentionGate(c_up, c_skip, max(c_skip // 2, 1))
        c_conv = c_up + c_skip if skip_mode in ("concat", "attention") else c_up
        self.conv = DoubleConv(c_conv, c_out, norm)

    def forward(self, x, skip):
        x = self.up(x)
        # Si la entrada no es múltiplo de 2^depth, el tamaño puede diferir en 1 px: se rellena.
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        if self.skip_mode == "concat":
            x = torch.cat([skip, x], dim=1)          # concatenar, no sumar (notas de clase §2.2)
        elif self.skip_mode == "attention":
            x = torch.cat([self.gate(x, skip), x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 2, n_out: int = 1, base: int = 32, depth: int = 4,
                 skip_mode: str = "concat", norm: str = "bn"):
        super().__init__()
        assert skip_mode in ("concat", "none", "attention")
        chs = [base * 2 ** i for i in range(depth + 1)]   # p. ej. 32, 64, 128, 256, 512
        self.depth = depth
        self.inc = DoubleConv(in_ch, chs[0], norm)
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1], norm) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([
            Up(chs[i + 1], chs[i], chs[i], skip_mode, norm) for i in reversed(range(depth))
        ])
        self.head = nn.Conv2d(chs[0], n_out, kernel_size=1)   # conv 1x1 -> logit por píxel
        self._init_weights()

    def _init_weights(self):
        # Inicialización de He (gaussiana con std sqrt(2/N)), como en el paper (notas §6).
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        x = skips.pop()                       # cuello de botella
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)                   # logits (B, 1, H, W); la sigmoide va en la pérdida


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for kw in [dict(), dict(depth=3), dict(skip_mode="none"), dict(skip_mode="attention")]:
        m = UNet(**kw)
        y = m(torch.randn(2, 2, 256, 256))
        print(kw, tuple(y.shape), f"{count_params(m):,} parámetros")
