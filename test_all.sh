#!/bin/bash
# Runs all splits for every experiment YAML passed as arguments.
# Usage: bash test_all.sh [rho|tau|f1] <config1.yaml> [config2.yaml ...]
# Example: bash test_all.sh rho configs/experiments/summe_v1.yaml configs/experiments/tvsum_v1.yaml
# Example all: bash test_all.sh rho configs/experiments/*.yaml

MONITOR=${1:-rho}
shift

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Usage: bash test_all.sh [rho|tau|f1] <config1.yaml> [config2.yaml ...]"
    exit 1
fi

for CONFIG in "${CONFIGS[@]}"; do
    echo "=============================="
    echo "Experiment: $CONFIG  [monitor: $MONITOR]"
    echo "=============================="
    bash test.sh "$CONFIG" "$MONITOR"
done
