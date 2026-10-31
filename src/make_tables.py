"""Junta los metrics.json de todas las corridas en tablas Markdown (media ± desv. entre semillas).

Convención de nombres: runs/<CONFIG>__s<SEMILLA>/eval_<split>/metrics.json
  python -m src.make_tables --runs runs > results/tablas.md
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    res = defaultdict(lambda: defaultdict(list))   # res[split][config] -> lista de dicts
    for f in glob.glob(os.path.join(a.runs, "*", "eval_*", "metrics.json")):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        split = os.path.basename(os.path.dirname(f)).replace("eval_", "")
        cfg = run.split("__s")[0]
        res[split][cfg].append(json.load(open(f)))
    fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}" if v else "—"
    for split in ["valid", "test", "bolivia", "peru"]:
        if split not in res:
            continue
        print(f"\n### Split: {split}\n")
        print("| Configuración | n semillas | IoU (pooled) | F1 | Precisión | Recall | mIoU por chip (sin vacíos) |")
        print("|---|---|---|---|---|---|---|")
        for cfg, lst in sorted(res[split].items()):
            g = lambda k: [m["pooled"][k] for m in lst]
            print(f"| {cfg} | {len(lst)} | {fmt(g('IoU'))} | {fmt(g('F1'))} | {fmt(g('precision'))} | "
                  f"{fmt(g('recall'))} | {fmt([m['mean_IoU_per_chip_excluding_empty'] for m in lst])} |")
    if "test" in res and ("bolivia" in res or "peru" in res):
        print("\n### Brecha de generalización (IoU pooled: test en distribución − región nueva)\n")
        print("| Configuración | test | bolivia | brecha | peru | brecha |")
        print("|---|---|---|---|---|---|")
        for cfg in sorted(res["test"]):
            t = np.mean([m["pooled"]["IoU"] for m in res["test"][cfg]])
            row = [f"{t:.3f}"]
            for sp in ["bolivia", "peru"]:
                if cfg in res.get(sp, {}):
                    v = np.mean([m["pooled"]["IoU"] for m in res[sp][cfg]])
                    row += [f"{v:.3f}", f"{t - v:+.3f}"]
                else:
                    row += ["—", "—"]
            print(f"| {cfg} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
