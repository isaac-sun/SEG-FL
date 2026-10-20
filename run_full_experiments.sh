#!/usr/bin/env bash
# Formal CSDD matrix, aligned with 20NEWS-FL / ECL-FL:
# 4 partitions x (no attack + DFR/SDFR/AFR x 10/30/50%) x 5 seeds = 200 runs,
# full participation, all five groupings in every run.
# Usage: FORCE_DEVICE=cuda bash run_full_experiments.sh [--require-cuda]
# PER_PARTITION_SUMMARY=1 refreshes <RESULTS_ROOT>/<partition>/ after each run
# and SKIP_FINAL_SUMMARY=1 skips the final merge (used by start.sh full-parallel).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
RESULTS_ROOT="${RESULTS_ROOT:-results/full_experiment}"
SEEDS="${SEEDS:-42,43,44,45,46}"
TASKS="${TASKS:-iid dirichlet:0.1 dirichlet:0.5 dirichlet:1.0}"
DEVICE="${FORCE_DEVICE:-auto}"

for TASK in $TASKS; do
    case "$TASK" in
        iid) ARGS=(--partition-method iid); ID=iid ;;
        dirichlet:*)
            ALPHA="${TASK#dirichlet:}"
            ARGS=(--partition-method dirichlet --dirichlet-alpha "$ALPHA")
            ID="dirichlet_alpha${ALPHA//./p}" ;;
        *) echo "Unsupported TASKS entry: $TASK" >&2; exit 2 ;;
    esac
    echo "===== SEG CSDD: $ID -> $RESULTS_ROOT/$ID ====="
    bash run.sh "$@" "${ARGS[@]}" \
        --results-dir "$RESULTS_ROOT/$ID" \
        --device "$DEVICE" \
        --num-rounds 25 --num-clients 20 --participation-ratio 1.0 \
        --local-epochs "${LOCAL_EPOCHS:-2}" --local-lr "${LOCAL_LR:-0.005}" \
        --attack-types dfr,sdfr,afr --malicious-ratios 0.1,0.3,0.5 \
        --experiment-seeds "$SEEDS" --num-mc-samples 20 \
        --summary-root "$([ "${PER_PARTITION_SUMMARY:-0}" = 1 ] && echo "$RESULTS_ROOT/$ID" || echo "$RESULTS_ROOT")"
done
[ "${SKIP_FINAL_SUMMARY:-0}" = 1 ] && exit 0
PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" summarize.py --results-root "$RESULTS_ROOT"
