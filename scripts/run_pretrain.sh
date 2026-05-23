#!/usr/bin/env bash
# Phase 2 launcher.
#
# Usage:
#   bash scripts/run_pretrain.sh configs/pretrain_summe.yaml
#
# Iterates over the `splits` list in the YAML. Each split is launched with the
# same config; results land under Summaries/<exp_name>/<dataset>/<dataset>_split<idx>/.

set -euo pipefail

CONFIG=${1:?"usage: $0 <config.yaml>"}
PY=${PYTHON:-python}

SPLITS=$($PY scripts/_yaml_get.py "$CONFIG" splits --default "0 1 2 3 4")
DEVICES=$($PY scripts/_yaml_get.py "$CONFIG" cuda_devices --default 0)

export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "[run_pretrain] config=$CONFIG splits='$SPLITS' CUDA_VISIBLE_DEVICES=$DEVICES"
for SPLIT in $SPLITS; do
    echo "==== split $SPLIT ===="
    $PY pretrain_aggregator.py --config "$CONFIG" --split_idx "$SPLIT"
done
