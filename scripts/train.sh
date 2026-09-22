#!/bin/bash
# The reference recipe on LFM2.5.  PY defaults to the pixi interpreter (`pixi run scripts/train.sh ...`).
#   scripts/train.sh full        2.6B, full typed decisions, staged core mixture (grad ckpt on)
#   scripts/train.sh core        1.2B, full typed decisions, staged core mixture
#   scripts/train.sh yesmom      Noul-only binary head on the 350M / 230M bases
#   scripts/train.sh delta M     continue from an existing checkpoint M on the new formats + a replay sample
set -e; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
PY=${PY:-python}; MODE=${1:-full}; INIT=${2:-}; OUT=${OUT:-runs/selectia_$MODE}; BASE=${BASE:-data/tasks.pkl}
mkdir -p data logs
[ -f "$BASE" ] || $PY -m selectia.data.core --out "$BASE"                # download + convert ~95 public datasets
MIX=${MIX:-data/mixture_$MODE.pkl}
[ -f "$MIX" ] || $PY -m selectia.data.mixture --base "$BASE" --mode "$MODE" --out "$MIX" --probes data/probes.pkl

EXTRA="--max_ctx 1536 --warmup 200 --max_tokens 16384 --accum 2 --none_prob 0.1 --schema_first_prob 0.5"
case "$MODE" in
  full)   INIT=${INIT:-LiquidAI/LFM2.5-2.6B-Base}; EXTRA="$EXTRA --max_options 255 --grad_ckpt";;
  core)   INIT=${INIT:-LiquidAI/LFM2.5-1.2B-Base}; EXTRA="$EXTRA --max_options 255";;
  yesmom) INIT=${INIT:-LiquidAI/LFM2.5-230M-Base}; EXTRA="--max_ctx 1536 --warmup 200 --max_tokens 16384 --accum 2 --none_prob 0 --schema_first_prob 0 --max_options 2 --yesmom";;
  delta)  EXTRA="$EXTRA --max_options 255"; [ -n "$INIT" ] || { echo "delta needs a checkpoint: scripts/train.sh delta M"; exit 2; };;
  *) echo "unknown mode $MODE (full|core|yesmom|delta)"; exit 2;;
esac
LR=1e-5; [ "$MODE" = delta ] && LR=8e-6
$PY -m selectia.train --model "$INIT" --data "$MIX" --out "$OUT" --epochs 1 --lr $LR \
    --eval_every 1000 --eval_limit 150 $EXTRA 2>&1 | tee "logs/train_$MODE.log"
scripts/evaluate.sh "$OUT/model"
