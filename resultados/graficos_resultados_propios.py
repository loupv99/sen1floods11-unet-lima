"""Gráficos de los resultados propios del grupo (Pregunta 2). Cada cifra viene de una corrida real:
- Colab T4 (Lu, 31-oct-2026), experimento de 3 semillas, configuración A_dorado: semillas 0 y 1 terminadas.
- Otsu-VH en valid/test/Bolivia: Colab (idéntico a lo medido en local).
- Lima: región construida en local (51 chips, Sentinel-1B 17-mar-2017 + EMSN038), línea base Otsu-VH.
- Con/sin saltos: corrida de verificación en CPU (U-Net base 16, 20 epochs, 1 semilla): modelo reducido.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
MAR, NAV, GRT, GRN = "#7A1518", "#1F3B66", "#8A8A8A", "#2E7D32"
plt.rcParams.update({"font.family": "Calibri", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def fuente(fig, txt):
    fig.text(0.01, 0.01, "Fuente: " + txt, fontsize=9, color="#5A5A5A", ha="left")


def etiquetas(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")


# 1) IoU: U-Net vs Otsu en test y Bolivia
unet = {"test": [0.6578, 0.6564], "bolivia": [0.6450, 0.6699]}
otsu = {"valid": 0.5067, "test": 0.5545, "bolivia": 0.7176}
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2); w = 0.36
m = [np.mean(unet["test"]), np.mean(unet["bolivia"])]
b1 = ax.bar(x - w / 2, m, w, color=NAV, label="U-Net A_dorado (media de 2 semillas)")
for i, k in enumerate(["test", "bolivia"]):
    ax.scatter([i - w / 2] * 2, unet[k], color="white", edgecolor="black", zorder=3, s=28, label="semillas 0 y 1" if i == 0 else None)
b2 = ax.bar(x + w / 2, [otsu["test"], otsu["bolivia"]], w, color=GRT, label="Otsu-VH (sin entrenar)")
for i, k in enumerate(["test", "bolivia"]):   # etiqueta por encima del punto más alto de las semillas
    ax.annotate(f"{m[i]:.3f}", (i - w / 2, max(unet[k])), ha="center", va="bottom", fontsize=10, xytext=(0, 6), textcoords="offset points")
etiquetas(ax, b2)
ax.set_xticks(x, ["Test (eventos vistos, 90 chips)", "Bolivia (región nueva, 15 chips)"])
ax.set_ylabel("IoU de la clase agua (pooled)"); ax.set_ylim(0, 0.85)
ax.set_title("La U-Net gana en eventos vistos (+0.10) y pierde en la región nueva (−0.06)", fontsize=13, color=MAR, weight="bold")
ax.legend(loc="upper center", fontsize=10, ncol=1, frameon=False)
fuente(fig, "Colab T4, corrida propia con 3 semillas (A_dorado: semillas 0 y 1 terminadas). Umbral elegido en validación.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/01_iou_unet_vs_otsu.png", dpi=170); plt.close(fig)

# 2) F1, precisión, recall en test y Bolivia
met = ["IoU", "F1", "Precisión", "Recall"]
u = {"test": np.mean([[0.6578, 0.7936, 0.7965, 0.7907], [0.6564, 0.7925, 0.7954, 0.7897]], 0),
     "bolivia": np.mean([[0.6450, 0.7842, 0.8604, 0.7204], [0.6699, 0.8023, 0.8508, 0.7591]], 0)}
o = {"test": [0.5545, 0.7134, 0.6820, 0.7478], "bolivia": [0.7176, 0.8356, 0.8035, 0.8704]}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, k, t in zip(axs, ["test", "bolivia"], ["Test (en distribución)", "Bolivia (región nueva)"]):
    x = np.arange(4)
    etiquetas(ax, ax.bar(x - w / 2, u[k], w, color=NAV, label="U-Net A_dorado")); etiquetas(ax, ax.bar(x + w / 2, o[k], w, color=GRT, label="Otsu-VH"))
    ax.set_xticks(x, met); ax.set_ylim(0, 1); ax.set_title(t, fontsize=13, color=MAR, weight="bold")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
axs[0].set_ylabel("valor de la clase agua")
fig.suptitle("En Bolivia la U-Net es más precisa (0.86) pero se le escapa agua: recall 0.74 vs 0.87 de Otsu", fontsize=13)
fuente(fig, "Colab T4, corrida propia (A_dorado, media de las semillas 0 y 1).")
fig.tight_layout(rect=(0, 0.04, 1, 0.86)); fig.savefig(f"{OUT}/02_metricas_test_bolivia.png", dpi=170); plt.close(fig)


# 3) Matrices de confusión medidas (píxeles)
def matriz(ax, tn, fp, fn, tp, titulo, cmap):
    M = np.array([[tn, fp], [fn, tp]], float); P = M / M.sum() * 100
    ax.imshow(P, cmap=cmap, vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{int(v):,}\n({P[i, j]:.1f} %)", ha="center", va="center", fontsize=11,
                color="white" if P[i, j] > 50 else "black")
    ax.set_xticks([0, 1], ["pred. no agua", "pred. agua"]); ax.set_yticks([0, 1], ["real no agua", "real agua"])
    ax.set_title(titulo, fontsize=12, color=MAR, weight="bold")


fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
matriz(axs[0], 17052124, 876827, 634364, 1880644, "Otsu-VH · test\nIoU 0.554", "Greys")
matriz(axs[1], 17369693, 559258, 526718, 1988290, "U-Net (verificación CPU) · test\nIoU 0.647", "Blues")
matriz(axs[2], 5103157, 5593108, 75248, 59613, "Otsu-VH · Lima\nIoU 0.010", "Reds")
fig.suptitle("Matrices de confusión medidas: la U-Net baja los falsos positivos un 36 %; en Lima, Otsu marca agua en el 52 % de la escena", fontsize=13)
fuente(fig, "test: corrida de verificación propia (CPU, U-Net reducida, 1 semilla); Lima: región construida por el grupo (51 chips).")
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(f"{OUT}/03_matrices_confusion.png", dpi=170); plt.close(fig)

# 4) Lima: la brecha
fig, ax = plt.subplots(figsize=(8, 4.8))
cats = ["Test", "Bolivia", "Lima"]; vals = [otsu["test"], otsu["bolivia"], 0.0104]
etiquetas(ax, ax.bar(cats, vals, 0.55, color=[GRT, GRT, MAR]))
ax.set_ylabel("IoU de agua · Otsu-VH"); ax.set_ylim(0, 0.85)
ax.set_title("Lima rompe incluso al método clásico: IoU 0.010", fontsize=13, color=MAR, weight="bold")
ax.text(2, 0.16, "umbral de la escena: −16.9 dB\n(eventos Sen1Floods11: −20 a −25 dB)\nOtsu separa desierto de ciudad,\nno agua de tierra", ha="center", fontsize=10)
fuente(fig, "región de Lima construida por el grupo: Sentinel-1B 17-mar-2017 + EMSN038 (51 chips, 1.25 % de agua).")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/04_lima_otsu.png", dpi=170); plt.close(fig)

# 5) Ablación de arquitectura (CPU, reducida)
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(2)
etiquetas(ax, ax.bar(x - w / 2, [0.647, 0.578], w, color=NAV, label="con saltos (concatenar)"))
etiquetas(ax, ax.bar(x + w / 2, [0.628, 0.624], w, color=SKY if (SKY := "#4A90C2") else None, label="sin saltos"))
ax.set_xticks(x, ["Test", "Bolivia"]); ax.set_ylim(0, 0.8); ax.set_ylabel("IoU de agua")
ax.set_title("Los saltos ayudan en eventos vistos (+0.02), no en la región nueva", fontsize=13, color=MAR, weight="bold")
ax.legend(frameon=False)
fuente(fig, "corrida de verificación propia en CPU: U-Net reducida (base 16), 20 epochs, 1 semilla. Indicativo, no definitivo.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/05_saltos_cpu.png", dpi=170); plt.close(fig)
print("ok")

# 7) Bono E6: fusión SAR + Sentinel-2 (misma U-Net reducida, 20 epochs, semilla 0)
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(2); w3 = 0.26
etiquetas(ax, ax.bar(x - w3, [0.647, 0.578], w3, color=NAV, label="Solo SAR (VV, VH)"))
etiquetas(ax, ax.bar(x, [0.8073, 0.7711], w3, color=GRN, label="SAR + Sentinel-2 (+6 bandas)"))
etiquetas(ax, ax.bar(x + w3, [otsu["test"], otsu["bolivia"]], w3, color=GRT, label="Otsu-VH"))
ax.set_xticks(x, ["Test", "Bolivia"]); ax.set_ylim(0, 1); ax.set_ylabel("IoU de agua")
ax.set_title("Añadir Sentinel-2 sube el IoU +0.16 en test y +0.19 en Bolivia", fontsize=13, color=MAR, weight="bold")
ax.legend(frameon=False, loc="upper right", fontsize=10)
fuente(fig, "corrida propia en CPU: U-Net reducida (base 16), 20 epochs, semilla 0; misma configuración en ambos modelos.")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/07_bono_sar_s2.png", dpi=170); plt.close(fig)
print("ok bono")
