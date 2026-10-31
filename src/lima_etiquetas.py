"""Etiquetas de referencia de Lima: geodatabase oficial de Copernicus EMS EMSN038 → GeoJSON en EPSG:4326.

- Inundación: capa `Potential_Maximum_flood` (producto "Flood extent delineation"; 13 polígonos, 13.58 km²,
  fuentes `river` y `maximun height`). Es la extensión MÁXIMA de la crecida, no el agua del día del radar.
- Zona rotulada: `EMSN038_AOI` ∩ huellas de las imágenes WorldView-2 (capa `Footprints`, 18 y 21-mar-2017).
  Fuera de esa zona no hay delineación, así que peru_prep la marca como -1 (sin dato).

Uso:  python -m src.lima_etiquetas --dir data/lima     (descarga ~2.3 GB la primera vez)
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile

URL = "https://mapping.emergency.copernicus.eu/download/EMSN038/EMSN038_Geodatabase.zip"


def main(argv=None):
    import geopandas as gpd

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/lima")
    a = ap.parse_args(argv)
    os.makedirs(a.dir, exist_ok=True)
    zp = os.path.join(a.dir, "EMSN038_Geodatabase.zip")
    gdb = os.path.join(a.dir, "EMSN038_UTM18S_v0.gdb")
    if not os.path.isdir(gdb):
        if not os.path.exists(zp):
            print("descargando", URL)
            urllib.request.urlretrieve(URL, zp + ".part"); os.replace(zp + ".part", zp)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():  # extrae solo la geodatabase y evita rutas fuera de la carpeta
                dst = os.path.normpath(os.path.join(a.dir, n))
                if n.startswith("EMSN038_UTM18S_v0.gdb/") and dst.startswith(os.path.normpath(a.dir)):
                    z.extract(n, a.dir)

    fl = gpd.read_file(gdb, layer="Potential_Maximum_flood").to_crs(4326)
    fl["geometry"] = fl.geometry.force_2d()
    fl[["source", "geometry"]].to_file(os.path.join(a.dir, "emsn038_flood.geojson"), driver="GeoJSON")

    aoi = gpd.read_file(gdb, layer="EMSN038_AOI").to_crs(32718)
    fp = gpd.read_file(gdb, layer="Footprints").to_crs(32718)
    zona = aoi.union_all().intersection(fp[fp["SDP"].str.contains("WorldView")].union_all())
    out = gpd.GeoDataFrame({"name": ["AOI EMSN038 ∩ WorldView-2 18/21-mar-2017"]}, geometry=[zona], crs=32718)
    out = out.to_crs(4326); out["geometry"] = out.geometry.force_2d()
    out.to_file(os.path.join(a.dir, "emsn038_aoi_wv2.geojson"), driver="GeoJSON")
    print(f"{len(fl)} polígonos de inundación ({fl.to_crs(32718).area.sum() / 1e6:.2f} km²); "
          f"zona rotulada {zona.area / 1e6:.1f} km²")


if __name__ == "__main__":
    main()
