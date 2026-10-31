"""Figura de la región de Lima: Sentinel-1 VH del 17-mar-2017 con los polígonos EMSN038 encima (entregable 4).

Uso:  python -m src.lima_mapa --s1 data/lima/lima_s1_20170317.tif --dir data/lima --out results/lima_mapa.png
"""
from __future__ import annotations

import argparse


def main(argv=None):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with rasterio.open(a.s1) as s:
        vh = s.read(2, out_shape=(s.height // 4, s.width // 4)); b = s.bounds
    fl = gpd.read_file(f"{a.dir}/emsn038_flood.geojson"); zona = gpd.read_file(f"{a.dir}/emsn038_aoi_wv2.geojson")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    vistas = [((b.left, b.right, b.bottom, b.top), "Lima norte · Sentinel-1B VH (17-mar-2017)"),
              ((-76.99, -76.84, -12.03, -11.93), "Zoom: Huachipa – Carapongo – Cajamarquilla")]
    for ax, ((x0, x1, y0, y1), tt) in zip(axs, vistas):
        ax.imshow(vh, cmap="gray", vmin=-26, vmax=-6, extent=(b.left, b.right, b.bottom, b.top))
        zona.boundary.plot(ax=ax, color="#F2C200", lw=1.2, ls="--")
        fl.plot(ax=ax, facecolor=(0.48, 0.08, 0.09, 0.55), edgecolor="#E53935", lw=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_title(tt, fontsize=12); ax.tick_params(labelsize=8)
    axs[0].legend(handles=[Patch(facecolor=(0.48, 0.08, 0.09, 0.55), edgecolor="#E53935", label="Inundación EMSN038"),
                           Line2D([], [], color="#F2C200", ls="--", label="Zona con WorldView-2 (18/21-mar)")],
                  loc="lower left", fontsize=9)
    plt.tight_layout(); plt.savefig(a.out, dpi=150)
    print("figura", a.out)


if __name__ == "__main__":
    main()
