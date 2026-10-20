#!/usr/bin/env bash
# Pilot: seed 42, IID and Dirichlet alpha=0.5, no attack + DFR/SDFR/AFR at 30%
# = 8 runs; every run computes all five groupings.  Use round_curves.csv
# (round_seconds, test_miou) to size the formal matrix and to check whether
# 2 local epochs converge too fast.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
RESULTS_ROOT="${RESULTS_ROOT:-results/pilot}"
DEVICE="${FORCE_DEVICE:-auto}"
COMMON=(--device "$DEVICE" --num-rounds 25 --num-clients 20 --participation-ratio 1.0
        --local-epochs "${LOCAL_EPOCHS:-2}" --local-lr "${LOCAL_LR:-0.005}"
        --attack-types dfr,sdfr,afr --malicious-ratios 0.3
        --experiment-seeds 42 --num-mc-samples 20 --summary-root "$RESULTS_ROOT")
bash run.sh "$@" "${COMMON[@]}" --partition-method iid --results-dir "$RESULTS_ROOT/iid"
bash run.sh "$@" "${COMMON[@]}" --partition-method dirichlet --dirichlet-alpha 0.5 \
    --results-dir "$RESULTS_ROOT/dirichlet_alpha0p5"
PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" summarize.py --results-root "$RESULTS_ROOT"
