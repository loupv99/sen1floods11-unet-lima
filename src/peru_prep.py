"""Construcción de la región de prueba de Lima (marzo 2017) con el MISMO formato que Sen1Floods11.

Fuente 1 (imagen): Sentinel-1 GRD IW, VV/VH en dB, 10 m, EPSG:4326. La forma más fiel de igualar el
preprocesamiento del dataset es exportarla desde Google Earth Engine (colección COPERNICUS/S1_GRD:
ruido térmico removido, calibración, corrección de terreno con SRTM, conversión a dB), que es la
misma plataforma con la que se construyó Sen1Floods11 (Bonafilia et al., 2020, §2.2). Ver
gee_export_lima.py. Escenas verificadas en ASF para un punto en Huachipa (-76.95, -11.98):
  S1B 2017-02-26, 03-05, 03-10, 03-17, 03-29, 04-03 (todas descendentes; órbitas relativas 69 y 171).
  La más cercana a las imágenes WorldView-2 de la delineación (18 y 21-mar) es la del 17-mar-2017.

Fuente 2 (etiquetas): producto "Flood extent delineation" de la activación EMSN038 (Copernicus EMS,
Risk & Recovery Mapping), descargado a mano como geodatabase o GeoJSON desde
https://mapping.emergency.copernicus.eu/activations/EMSN038/

Codificación de la etiqueta rasterizada:
   1 = dentro de un polígono de inundación;
   0 = dentro del área de interés (AOI) del producto pero fuera de los polígonos;
  -1 = fuera del AOI, sin dato SAR, o dentro de las máscaras opcionales (p. ej., el cauce permanente
       del río, que es agua pero el producto no lo marca como inundación).

Uso (requiere rasterio, geopandas, shapely):
  python -m src.peru_prep --s1 lima_s1_20170317.tif --flood emsn038_flood.geojson \
      --aoi emsn038_aoi.geojson [--mask cauce_rimac_chillon.geojson] --root data/sen1floods11/v1.1
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def main(argv=None):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, transform as win_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="GeoTIFF de 2 bandas (VV, VH) en dB, EPSG:4326, 10 m")
    ap.add_argument("--flood", required=True, help="polígonos de inundación EMSN038 (GeoJSON o .gdb)")
    ap.add_argument("--flood-layer", default=None, help="capa dentro del .gdb, si aplica")
    ap.add_argument("--flood-query", default=None,
                    help="filtro pandas sobre atributos, p. ej. \"notation == 'Flooded area'\" (POR CONFIRMAR "
                         "el nombre real del campo al abrir el producto)")
    ap.add_argument("--aoi", default=None, help="polígono del área analizada por EMS (sin él: todo es 0/1)")
    ap.add_argument("--mask", default=None, help="polígonos a ignorar (-1), p. ej. cauce permanente")
    ap.add_argument("--root", required=True)
    ap.add_argument("--chip", type=int, default=512)
    ap.add_argument("--min-valid", type=float, default=0.5, help="fracción mínima de píxeles con dato")
    args = ap.parse_args(argv)

    out_img = os.path.join(args.root, "peru", "S1Hand")
    out_lab = os.path.join(args.root, "peru", "LabelHand")
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lab, exist_ok=True)

    with rasterio.open(args.s1) as src:
        assert src.count >= 2, "se esperan 2 bandas: VV, VH"
        assert src.crs.to_epsg() == 4326, f"CRS {src.crs}: reproyectar a EPSG:4326 como Sen1Floods11"
        flood = gpd.read_file(args.flood, layer=args.flood_layer).to_crs(4326)
        if args.flood_query:
            flood = flood.query(args.flood_query)
        aoi = gpd.read_file(args.aoi).to_crs(4326) if args.aoi else None
        mask = gpd.read_file(args.mask).to_crs(4326) if args.mask else None
        print(f"escena {src.width}x{src.height} px, {len(flood)} polígonos de inundación")
        rows, k = [], 0
        for r0 in range(0, src.height - args.chip + 1, args.chip):
            for c0 in range(0, src.width - args.chip + 1, args.chip):
                win = Window(c0, r0, args.chip, args.chip)
                x = src.read([1, 2], window=win).astype(np.float32)
                ok = np.isfinite(x).all(0) & (x != 0).all(0)      # 0 exacto = relleno de exportación
                if ok.mean() < args.min_valid:
                    continue
                tr = win_transform(win, src.transform)
                shp = (args.chip, args.chip)
                y = np.zeros(shp, np.int16)
                if aoi is not None:
                    inside = rasterize(((g, 1) for g in aoi.geometry), out_shape=shp, transform=tr, fill=0, dtype="uint8")
                    if inside.sum() == 0:
                        continue
                    y[inside == 0] = -1
                fl = rasterize(((g, 1) for g in flood.geometry), out_shape=shp, transform=tr, fill=0, dtype="uint8")
                y[(fl == 1) & (y != -1)] = 1
                if mask is not None:
                    mk = rasterize(((g, 1) for g in mask.geometry), out_shape=shp, transform=tr, fill=0, dtype="uint8")
                    y[mk == 1] = -1
                y[~ok] = -1
                x[:, ~ok] = np.nan
                cid = f"Peru_{k:04d}"
                prof = dict(driver="GTiff", width=args.chip, height=args.chip, crs="EPSG:4326", transform=tr, compress="deflate")
                with rasterio.open(os.path.join(out_img, f"{cid}_S1Hand.tif"), "w", count=2, dtype="float32", **prof) as d:
                    d.write(x)
                with rasterio.open(os.path.join(out_lab, f"{cid}_LabelHand.tif"), "w", count=1, dtype="int16", **prof) as d:
                    d.write(y[None])
                rows.append((f"{cid}_S1Hand.tif", f"{cid}_LabelHand.tif", int((y == 1).sum()), int((y == 0).sum())))
                k += 1
    with open(os.path.join(args.root, "peru", "flood_peru_data.csv"), "w", newline="") as f:
        csv.writer(f).writerows([r[:2] for r in rows])
    agua = sum(r[2] for r in rows); seco = sum(r[3] for r in rows)
    print(f"{len(rows)} chips de Lima escritos; píxeles inundados {agua:,} ({100 * agua / max(agua + seco, 1):.2f} % de los válidos)")


if __name__ == "__main__":
    main()
