"""Descarga de Sen1Floods11 v1.1 desde el bucket público (sin cuenta), solo las carpetas necesarias.

  python -m src.download --dest data/sen1floods11 [--with-weak] [--with-s2]

Tamaños medidos al listar el bucket (2026-10-31): S1Hand 0.73 GB, S2Hand 1.02 GB, S1Weak 7.32 GB,
etiquetas < 0.05 GB. S2Weak (25.7 GB) NO se descarga: la red solo recibe SAR; las etiquetas débiles
de Sentinel-2 ya están en S2IndexLabelWeak.
"""
import argparse
import concurrent.futures as cf
import json
import os
import urllib.parse
import urllib.request

BUCKET = "sen1floods11"
API = "https://storage.googleapis.com/storage/v1/b/{b}/o?{q}"
GET = "https://storage.googleapis.com/{b}/{name}"


def ls(prefix):
    out, tok = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if tok:
            q["pageToken"] = tok
        d = json.load(urllib.request.urlopen(API.format(b=BUCKET, q=urllib.parse.urlencode(q))))
        out += [i["name"] for i in d.get("items", [])]
        tok = d.get("nextPageToken")
        if not tok:
            return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/sen1floods11")
    ap.add_argument("--with-weak", action="store_true")
    ap.add_argument("--with-s2", action="store_true", help="S2Hand (para figuras y para el bono)")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    hand = "v1.1/data/flood_events/HandLabeled/"
    weak = "v1.1/data/flood_events/WeaklyLabeled/"
    prefixes = ["v1.1/splits/", "v1.1/Sen1Floods11_Metadata.geojson",
                hand + "S1Hand/", hand + "LabelHand/", hand + "S1OtsuLabelHand/", hand + "JRCWaterHand/"]
    if a.with_s2:
        prefixes.append(hand + "S2Hand/")
    if a.with_weak:
        prefixes += [weak + "S1Weak/", weak + "S1OtsuLabelWeak/", weak + "S2IndexLabelWeak/"]
    names = [n for p in prefixes for n in ls(p) if not n.endswith("/") and not n.endswith(".json")]

    def get(name):
        dst = os.path.join(a.dest, name)
        if os.path.exists(dst):
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(GET.format(b=BUCKET, name=urllib.parse.quote(name)), dst + ".part")
        os.replace(dst + ".part", dst)
        return 1

    with cf.ThreadPoolExecutor(a.threads) as ex:
        n = sum(ex.map(get, names))
    print(f"{len(names)} archivos en el listado, {n} descargados ahora -> {os.path.join(a.dest, 'v1.1')}")


if __name__ == "__main__":
    main()
