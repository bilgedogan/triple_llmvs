#!/bin/bash
# Usage: bash test.sh <config_path> [extra args]

CONFIG=${1:?"Usage: bash test.sh <config_path> [extra args]"}
shift

EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")
mkdir -p "Summaries/${EXP_NAME}"

for MONITOR in rho tau f1; do
    echo ""
    echo "=============================="
    echo "Monitor: $MONITOR — $EXP_NAME"
    echo "=============================="

    # Write header (overwrite any previous run for this monitor)
    python - "$EXP_NAME" "$MONITOR" <<'PYEOF'
import sys
from datetime import datetime
exp, monitor = sys.argv[1], sys.argv[2]
LABELS = ['F1','kTau','sRho','mp15','mAP50']
C = 9
header = f"{'Split':<6}" + ''.join(f'{l:>{C}}' for l in LABELS)
sep    = '-' * len(header)
lines  = [
    f'Experiment : {exp}',
    f'Monitor    : {monitor}',
    f'Generated  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    '', header, sep,
]
with open(f'Summaries/{exp}/best_{monitor}_results.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
PYEOF

    ALL_PASS=true
    for SPLIT in 0 1 2 3 4; do
        CKPT_DIR="Summaries/${EXP_NAME}/split_${SPLIT}/best_${MONITOR}_model"
        WEIGHTS=$(find "$CKPT_DIR" -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null | sort | tail -1)

        if [ -z "$WEIGHTS" ]; then
            echo "[SKIP] split $SPLIT — no checkpoint in $CKPT_DIR"
            ALL_PASS=false
            continue
        fi

        echo "--- split $SPLIT: $WEIGHTS"
        CUDA_VISIBLE_DEVICES=0 python test.py \
            --config "$CONFIG" \
            --split_idx $SPLIT \
            --weights "$WEIGHTS" \
            --monitor "$MONITOR" \
            "$@"
    done

    [ "$ALL_PASS" = false ] && echo "WARNING: some splits skipped for monitor=$MONITOR"

    # Append mean/std
    python - "$EXP_NAME" "$MONITOR" <<'PYEOF'
import sys, numpy as np
exp, monitor = sys.argv[1], sys.argv[2]
KEYS = ['val_f1','val_kTau','val_sRho','val_mp15','val_map50']
C    = 9

out = f'Summaries/{exp}/best_{monitor}_results.txt'
with open(out) as f:
    lines = f.readlines()

data_rows = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
if not data_rows:
    print('No split results to aggregate.'); sys.exit(0)

vals = {k: [] for k in KEYS}
for row in data_rows:
    parts = row.split()
    for i, k in enumerate(KEYS):
        try: vals[k].append(float(parts[i+1]))
        except: pass

fmt   = lambda v: f'{v:.4f}' if v == v else '  N/A'
sep   = '-' * (6 + C * len(KEYS))
means = {k: np.nanmean(v) for k, v in vals.items()}
stds  = {k: np.nanstd(v)  for k, v in vals.items()}

with open(out, 'a') as f:
    f.write(sep + '\n')
    f.write(f"{'Mean':<6}" + ''.join(f'{fmt(means[k]):>{C}}' for k in KEYS) + '\n')
    f.write(f"{'Std':<6}"  + ''.join(f'{fmt(stds[k]):>{C}}'  for k in KEYS) + '\n')
    f.write(sep + '\n')

with open(out) as f: print(f.read())
print(f'Saved → {out}')
PYEOF

done
