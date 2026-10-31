#!/usr/bin/env bash
# Único comando de extremo a extremo (lo exige el enunciado):   bash run_all.sh
# Variables opcionales:  SEEDS="0 1 2"   QUICK=1 (prueba de humo: 2 epochs, pocos chips)   PERU=1
set -euo pipefail
cd "$(dirname "$0")"
ROOT=data/sen1floods11/v1.1
SEEDS=${SEEDS:-"0 1 2"}
EP_HAND=${EP_HAND:-60}; EP_WEAK=${EP_WEAK:-15}; EP_FT=${EP_FT:-40}
EXTRA=""
if [[ "${QUICK:-0}" == "1" ]]; then
  SEEDS="0"; EP_HAND=2; EP_WEAK=1; EP_FT=1
  EXTRA="--max-train-chips 32 --max-val-chips 8 --max-weak 64"
fi

# 1) datos (idempotente: no vuelve a bajar lo que ya está)
python -m src.download --dest data/sen1floods11 --with-weak --with-s2

# 2) línea base clásica (Otsu sobre VH, la que trae el dataset)
for sp in valid test bolivia; do python -m src.evaluate --root $ROOT --baseline otsu --split $sp; done

# Idempotente: si una corrida ya terminó (existe train_log.json) no se reentrena; así se puede
# relanzar tras una desconexión de Colab o con PERU=1 sin repetir horas de GPU.
fit () {   # $1 = nombre de corrida, resto = argumentos de train.py
  local run=$1; shift
  [[ -f "runs/$run/train_log.json" ]] || python -m src.train --root $ROOT --run "$run" $EXTRA "$@"
}
ev () {    # $1 = nombre de corrida. Umbral elegido en VALID y aplicado sin cambios a test/bolivia.
  for sp in valid test bolivia; do
    [[ -f "runs/$1/eval_$sp/metrics.json" ]] || python -m src.evaluate --root $ROOT --ckpt "runs/$1/best.pt" --split $sp --tune-threshold
  done
}
train () {  # $1 = nombre de configuración, resto = argumentos de train.py
  local cfg=$1; shift
  for s in $SEEDS; do fit "${cfg}__s${s}" --seed $s "$@"; ev "${cfg}__s${s}"; done
}

# 3) Ablación (a): fuente de supervisión  (pérdida combinada, U-Net prof. 4, saltos por concatenación)
train A_dorado           --train-data hand                  --loss combo --epochs $EP_HAND
train B_dorado+debilS1   --train-data hand+weak_s1 --weak-weight 0.5 --hand-oversample 4 --loss combo --epochs $EP_WEAK
train C_dorado+debilS2   --train-data hand+weak_s2 --weak-weight 0.5 --hand-oversample 4 --loss combo --epochs $EP_WEAK
# D: preentrenar con débil S2 y ajustar con dorado (dos etapas, misma semilla)
for s in $SEEDS; do
  fit "D_pre_debilS2__s${s}" --seed $s --train-data weak_s2 --loss combo --epochs $EP_WEAK
  fit "D_debilS2_a_dorado__s${s}" --seed $s --train-data hand --loss combo --epochs $EP_FT --lr 2e-4       --init "runs/D_pre_debilS2__s${s}/best.pt"
  ev "D_debilS2_a_dorado__s${s}"
done

# 4) Ablación (b): función de pérdida (datos dorados)  -- A_dorado ya es la combinada
train L_bce   --train-data hand --loss bce  --epochs $EP_HAND
train L_dice  --train-data hand --loss dice --epochs $EP_HAND

# 5) Ablación (b'): arquitectura (datos dorados, pérdida combinada)
train R_sin_saltos --train-data hand --loss combo --skip-mode none      --epochs $EP_HAND
train R_atencion   --train-data hand --loss combo --skip-mode attention --epochs $EP_HAND
train R_prof3      --train-data hand --loss combo --depth 3             --epochs $EP_HAND

# 6) Bono: fusión SAR + Sentinel-2 (solo chips dorados; las nubes ya son -1 en la etiqueta)
train X_sar+s2 --train-data hand --loss combo --use-s2 --epochs $EP_HAND

# 7) Región de prueba de Lima (entregable 4): Sentinel-1B del 17-mar-2017 (Planetary Computer, sin cuenta)
#    + delineación EMSN038 (Copernicus EMS). Se construye una sola vez y se evalúa sin reentrenar.
if [[ "${PERU:-0}" == "1" ]]; then
  if [[ ! -f $ROOT/peru/flood_peru_data.csv ]]; then
    python -m src.lima_etiquetas --dir data/lima
    python -m src.lima_s1 --out data/lima/lima_s1_20170317.tif
    python -m src.peru_prep --s1 data/lima/lima_s1_20170317.tif --flood data/lima/emsn038_flood.geojson \
      --aoi data/lima/emsn038_aoi_wv2.geojson --root $ROOT
    python -m src.lima_otsu --root $ROOT
  fi
  mkdir -p results
  python -m src.lima_mapa --s1 data/lima/lima_s1_20170317.tif --dir data/lima --out results/lima_mapa.png || true
  python -m src.evaluate --root $ROOT --baseline otsu --split peru
  for d in runs/*__s*/; do [[ -f "$d/best.pt" && "$d" != *D_pre_* && "$d" != *X_sar+s2* ]] && \
    python -m src.evaluate --root $ROOT --ckpt "$d/best.pt" --split peru --tune-threshold; done
fi

# 8) Análisis de fallos del mejor modelo (cambiar BEST si otro gana en VALID)
BEST=${BEST:-A_dorado__s0}
python -m src.failure_analysis --root $ROOT --eval-dir "runs/$BEST/eval_test"
python -m src.failure_analysis --root $ROOT --eval-dir "runs/$BEST/eval_bolivia"
[[ -f "runs/$BEST/eval_peru/probs.npz" ]] && python -m src.failure_analysis --root $ROOT --eval-dir "runs/$BEST/eval_peru"

# 9) Tablas
mkdir -p results
python -m src.make_tables --runs runs | tee results/tablas.md
