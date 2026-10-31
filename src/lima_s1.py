"""Imagen Sentinel-1 de Lima (17-mar-2017) en el formato de Sen1Floods11, sin cuenta de Earth Engine.

Fuente: Microsoft Planetary Computer (acceso anónimo), escena
  S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848_004747_0084AE  (descendente, órbita relativa 171)
colección `sentinel-1-rtc` (γ0 con terreno aplanado, 10 m, UTM 18S) + `sentinel-1-grd` (solo el XML de anotación,
para leer el ángulo de incidencia de la grilla de geolocalización).

Sen1Floods11 usa COPERNICUS/S1_GRD de Earth Engine: σ0 en dB con corrección geométrica (no radiométrica) de terreno.
En terreno plano σ0 = γ0 · cos(θ), con θ el ángulo de incidencia sobre el elipsoide, así que aquí se aplica
σ0_dB = γ0_dB + 10·log10(cos θ) con θ interpolado de la anotación del producto. En laderas la conversión no es exacta
(el RTC además corrige el área iluminada): es una limitación documentada, no un ajuste fino.

Salida: GeoTIFF de 2 bandas float32 (VV, VH en dB), EPSG:4326, píxel 8.98315e-05° (el de los chips de Sen1Floods11),
NaN donde no hay dato. Uso:
  python -m src.lima_s1 --out lima_s1_20170317.tif [--bbox -77.23 -12.10 -76.70 -11.67]   (cubre el AOI de EMSN038)
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.interpolate import griddata

SCENE = "S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848_004747_0084AE"
PIX = 8.98315e-05  # grados, medido en los chips S1Hand de Sen1Floods11


def incidence_grid(grd_item, lons, lats):
    """Interpola el ángulo de incidencia (grados) de la anotación del GRD sobre la grilla de salida."""
    xml = requests.get(grd_item.assets["schema-product-vv"].href, timeout=120).text
    pts = ET.fromstring(xml).findall(".//geolocationGridPoint")
    lon = np.array([float(p.find("longitude").text) for p in pts])
    lat = np.array([float(p.find("latitude").text) for p in pts])
    inc = np.array([float(p.find("incidenceAngle").text) for p in pts])
    LON, LAT = np.meshgrid(lons, lats)
    return griddata((lon, lat), inc, (LON, LAT), method="linear"), (inc.min(), inc.max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, default=[-77.23, -12.10, -76.70, -11.67],
                    metavar=("W", "S", "E", "N"))
    a = ap.parse_args(argv)
    W, S, E, N = a.bbox

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    rtc = cat.get_collection("sentinel-1-rtc").get_item(SCENE + "_rtc")
    grd = cat.get_collection("sentinel-1-grd").get_item(SCENE)

    width, height = int(round((E - W) / PIX)), int(round((N - S) / PIX))
    dst_tr = from_origin(W, N, PIX, PIX)
    out = np.full((2, height, width), np.nan, np.float32)
    for b, pol in enumerate(["vv", "vh"]):
        with rasterio.open(rtc.assets[pol].href) as src:
            l, bt, r, t = transform_bounds("EPSG:4326", src.crs, W, S, E, N, densify_pts=21)
            win = from_bounds(l - 200, bt - 200, r + 200, t + 200, src.transform).round_offsets().round_lengths()
            g0 = src.read(1, window=win).astype(np.float32)
            g0[(g0 == src.nodata) | (g0 <= 0)] = np.nan
            reproject(g0, out[b], src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=dst_tr, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.bilinear)
        print(f"{pol}: {np.isfinite(out[b]).mean():.1%} de píxeles con dato")

    lons = W + PIX * (np.arange(width) + 0.5)
    lats = N - PIX * (np.arange(height) + 0.5)
    theta, (tmin, tmax) = incidence_grid(grd, lons, lats)
    print(f"ángulo de incidencia en el recorte: {np.nanmin(theta):.2f}°–{np.nanmax(theta):.2f}° "
          f"(escena completa {tmin:.2f}°–{tmax:.2f}°)")
    corr = 10 * np.log10(np.cos(np.deg2rad(theta))).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(out) + corr[None]
    print(f"corrección γ0→σ0 media: {np.nanmean(corr):.2f} dB")
    for b, pol in enumerate(["VV", "VH"]):
        print(f"{pol} σ0 dB: media {np.nanmean(db[b]):.2f}  desv. {np.nanstd(db[b]):.2f}")

    prof = dict(driver="GTiff", width=width, height=height, count=2, dtype="float32", crs="EPSG:4326",
                transform=dst_tr, nodata=np.nan, compress="deflate", tiled=True)
    with rasterio.open(a.out, "w", **prof) as d:
        d.write(db.astype(np.float32))
        d.update_tags(source=SCENE, conversion="sigma0_dB = gamma0_RTC_dB + 10log10(cos(theta_ellipsoid))")
    print(f"escrito {a.out}: {width}x{height} px")


if __name__ == "__main__":
    main()
