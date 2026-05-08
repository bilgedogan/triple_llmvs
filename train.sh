#!/bin/bash
# Usage: bash train.sh <config_path> [extra args]
# Example: bash train.sh configs/experiments/summe_v1.yaml
# Example with override: bash train.sh configs/experiments/summe_v1.yaml --num_heads 8

CONFIG=${1:?"Usage: bash train.sh <config_path> [extra args]"}
shift  # remaining args passed through to train.py

for SPLIT in 0 1 2 3 4; do
    echo "=============================="
    echo "Training split $SPLIT — config: $CONFIG"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config "$CONFIG" \
        --split_idx $SPLIT \
        "$@"
done
