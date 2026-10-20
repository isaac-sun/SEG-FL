#!/usr/bin/env bash
# Learning-rate probe before the pilot: no attack, no Shapley, IID, seed 42,
# 25 rounds, full participation, one run per learning rate (LR_LIST).
# lr_probe_report.py then shows per learning rate how much of the loss drop
# happens in round 0 and the final test mIoU, and suggests a value for
# LOCAL_LR (pilot and full).  The report is also saved to
# <RESULTS_ROOT>/lr_probe_report.txt (read it with cat; no Python packages needed).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
RESULTS_ROOT="${RESULTS_ROOT:-results/lr_probe}"
DEVICE="${FORCE_DEVICE:-auto}"
LR_LIST="${LR_LIST:-0.01 0.005 0.0025 0.001}"
for LR in $LR_LIST; do
    bash run.sh "$@" --device "$DEVICE" --partition-method iid \
        --num-rounds 25 --num-clients 20 --participation-ratio 1.0 \
        --local-epochs "${LOCAL_EPOCHS:-2}" --local-lr "$LR" \
        --attack-types "" --experiment-seeds 42 --no-shapley --no-plots \
        --results-dir "$RESULTS_ROOT/lr$LR" --summary-root "$RESULTS_ROOT/lr$LR"
done
PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" \
    lr_probe_report.py "$RESULTS_ROOT" | tee "$RESULTS_ROOT/lr_probe_report.txt"
