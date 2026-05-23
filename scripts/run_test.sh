#!/usr/bin/env bash
# Phase 5 (test) launcher.
#
# Usage:
#   bash scripts/run_test.sh configs/test_summe.yaml
#
# test_rl.py iterates over `splits` from the YAML internally, substituting
# {split} in the `weights` template, and writes a single results.txt under
# Summaries/<exp_name>/<dataset>/ (or `result_dir` if set in YAML).

set -euo pipefail

CONFIG=${1:?"usage: $0 <config.yaml>"}
PY=${PYTHON:-python}

DEVICES=$($PY scripts/_yaml_get.py "$CONFIG" cuda_devices --default 0)
export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "[run_test] config=$CONFIG CUDA_VISIBLE_DEVICES=$DEVICES"
$PY test_rl.py --config "$CONFIG"
