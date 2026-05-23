#!/usr/bin/env bash
# Phase 3 (RL/PPO) launcher.
#
# Usage:
#   bash scripts/run_rl.sh configs/rl_summe.yaml
#
# YAML may contain `pretrained_aggregator: .../summe_split{split}/...` — the
# {split} placeholder is substituted per loop iteration so a single config can
# cover all 5 folds. Append `--joint_finetune` via the YAML to enable Phase 4.

set -euo pipefail

CONFIG=${1:?"usage: $0 <config.yaml>"}
PY=${PYTHON:-python}

SPLITS=$($PY scripts/_yaml_get.py "$CONFIG" splits --default "0 1 2 3 4")
DEVICES=$($PY scripts/_yaml_get.py "$CONFIG" cuda_devices --default 0)
PRETRAIN_TEMPLATE=$($PY scripts/_yaml_get.py "$CONFIG" pretrained_aggregator --default "")

export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "[run_rl] config=$CONFIG splits='$SPLITS' CUDA_VISIBLE_DEVICES=$DEVICES"
for SPLIT in $SPLITS; do
    echo "==== split $SPLIT ===="
    EXTRA=()
    if [[ -n "$PRETRAIN_TEMPLATE" && "$PRETRAIN_TEMPLATE" == *"{split}"* ]]; then
        CKPT="${PRETRAIN_TEMPLATE//\{split\}/$SPLIT}"
        EXTRA=(--pretrained_aggregator "$CKPT")
    fi
    $PY train_rl.py --config "$CONFIG" --split_idx "$SPLIT" "${EXTRA[@]}"
done
