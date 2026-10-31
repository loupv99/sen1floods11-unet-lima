"""Exporta desde Google Earth Engine la escena Sentinel-1 de Lima en el formato de Sen1Floods11.

Requiere una cuenta de Earth Engine con un proyecto de Google Cloud (gratis para uso académico):
  pip install earthengine-api && earthengine authenticate
  python -m src.gee_export_lima --project MI_PROYECTO --date 2017-03-17
El GeoTIFF aparece en Google Drive (carpeta "sen1floods11_peru"); se baja y se pasa a peru_prep.py.

COPERNICUS/S1_GRD ya viene en dB (10*log10 de sigma0), con ruido térmico removido, calibrado y con
corrección de terreno: es el mismo producto que usaron los autores de Sen1Floods11.
"""
import argparse

# AOI que cubre Carapongo/Huachipa, Cajamarquilla, San Martín de Porres, Comas, Carabayllo y Puente
# Piedra. Coordenadas aproximadas [lon_min, lat_min, lon_max, lat_max]; ajustar al AOI del producto EMS.
LIMA_AOI = [-77.15, -12.05, -76.80, -11.78]


def main():
    import ee
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--date", default="2017-03-17", help="día de la escena (UTC)")
    ap.add_argument("--orbit", default="", help="órbita relativa (69 o 171) si se quiere una sola")
    args = ap.parse_args()
    ee.Initialize(project=args.project)
    aoi = ee.Geometry.Rectangle(LIMA_AOI)
    d0 = ee.Date(args.date)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(d0, d0.advance(1, "day"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    if args.orbit:
        col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", int(args.orbit)))
    print("escenas encontradas:", col.size().getInfo(),
          col.aggregate_array("system:index").getInfo())
    img = col.select(["VV", "VH"]).mosaic().clip(aoi).toFloat()
    # Ángulo de incidencia medio sobre el AOI: dato para la hipótesis de generalización (entrega 4).
    ang = col.select("angle").mosaic().reduceRegion(ee.Reducer.mean(), aoi, 100).getInfo()
    print("ángulo de incidencia medio (grados):", ang)
    task = ee.batch.Export.image.toDrive(image=img, description=f"lima_s1_{args.date.replace('-', '')}",
                                         folder="sen1floods11_peru", region=aoi, crs="EPSG:4326",
                                         scale=10, maxPixels=1e10, fileFormat="GeoTIFF")
    task.start()
    print("exportación lanzada:", task.status())


if __name__ == "__main__":
    main()
