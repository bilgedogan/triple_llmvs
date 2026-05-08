#!/bin/bash
# Runs all splits for every experiment YAML passed as arguments.
# Usage: bash train_all.sh configs/experiments/summe_v1.yaml configs/experiments/tvsum_v1.yaml
# Or all experiments at once: bash train_all.sh configs/experiments/*.yaml

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Usage: bash train_all.sh <config1.yaml> [config2.yaml ...]"
    exit 1
fi

for CONFIG in "${CONFIGS[@]}"; do
    echo "=============================="
    echo "Experiment: $CONFIG"
    echo "=============================="
    bash train.sh "$CONFIG"
done
