#!/usr/bin/env bash
# One-command launcher for the SEG-FL experiments.
#
#   bash start.sh prepare      # download / build VOC 2012 aug arrays and check weights
#   bash start.sh pilot        # pilot: seed 42, IID + alpha=0.5, no attack + 30% attacks (default)
#   bash start.sh full         # 4 partitions x 10 conditions x 5 seeds = 200 runs
#   bash start.sh full-parallel  # the same 200 runs, one process per partition at once
#   bash start.sh all          # pilot -> full
#   bash start.sh status       # show whether a run is active and the log tail
#   bash start.sh stop         # stop the active run (resume later with the same command)
#
# Runs in the background with nohup and logs to logs/<mode>_<time>.log.
# Re-running the same mode resumes: finished runs are skipped and an
# interrupted run continues from its last finished round.
#
# Environment overrides: FORCE_DEVICE (default cuda), SEEDS, TASKS,
# RESULTS_ROOT, PYTHON_BIN, PIP_INDEX_URL.  FOREGROUND=1 runs in the terminal.
# full-parallel: THREADS_PER_RUN (default: CPU cores / number of partitions).
# LOCAL_EPOCHS (default 2) is passed to pilot and full.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
MODE="${1:-pilot}"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/run.pid"
CURRENT="$LOG_DIR/current_logs.txt"   # logs of the most recent launch
mkdir -p "$LOG_DIR"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
kill_tree() {   # stop a process and all of its descendants
    local child
    for child in $(pgrep -P "$1" 2>/dev/null || true); do kill_tree "$child"; done
    kill -TERM "$1" 2>/dev/null || true
}

case "$MODE" in
    status)
        if running; then echo "running (pid $(cat "$PID_FILE"))"; else echo "not running"; fi
        if [ -f "$CURRENT" ] && [ "$(wc -l < "$CURRENT")" -gt 1 ]; then
            while read -r log; do   # parallel launch: last progress lines per log
                echo "== $log"
                grep -E "round|done|skip|resume|Error|Traceback|failed|finished" "$log" \
                    2>/dev/null | tail -n 3 || true
            done < "$CURRENT"
        else
            LATEST="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)"
            [ -n "$LATEST" ] && { echo "== $LATEST"; tail -n 20 "$LATEST"; }
        fi
        exit 0 ;;
    stop)
        if running; then
            PID="$(cat "$PID_FILE")"
            kill_tree "$PID"
            echo "stopped pid $PID; run 'bash start.sh <mode>' again to resume"
        else
            echo "not running"
        fi
        rm -f "$PID_FILE"
        exit 0 ;;
    prepare)  STEPS=(fetch_data.sh) ;;
    pilot)    STEPS=(run_pilot.sh) ;;
    full)     STEPS=(run_full_experiments.sh) ;;
    full-parallel) STEPS=() ;;
    all)      STEPS=(run_pilot.sh run_full_experiments.sh) ;;
    -h|--help|help) sed -n 2,17p "$0"; exit 0 ;;
    *) echo "unknown mode: $MODE (prepare|pilot|full|full-parallel|all|status|stop)" >&2; exit 2 ;;
esac

if running; then
    echo "another run is active (pid $(cat "$PID_FILE")); use 'bash start.sh stop' first" >&2
    exit 1
fi

export FORCE_DEVICE="${FORCE_DEVICE:-cuda}"
EXTRA=()
if [ "$FORCE_DEVICE" = "cuda" ]; then EXTRA=(--require-cuda); fi
LOG="${SEG_LOG:-$LOG_DIR/${MODE}_$(date +%Y%m%d_%H%M%S).log}"
export SEG_LOG="$LOG"
PARALLEL_TASKS="${TASKS:-iid dirichlet:0.1 dirichlet:0.5 dirichlet:1.0}"
partition_dir() { case "$1" in iid) echo iid ;; dirichlet:*) a="${1#dirichlet:}"; echo "dirichlet_alpha${a//./p}" ;; esac; }

job() {
    for step in "${STEPS[@]}"; do
        echo "===== $(date '+%F %T') start $step ====="
        if [ "$step" = fetch_data.sh ]; then set -- ; else set -- ${EXTRA[@]+"${EXTRA[@]}"}; fi
        bash "$step" "$@" || { echo "===== $step failed; fix and rerun to resume ====="; return 1; }
        echo "===== $(date '+%F %T') finished $step ====="
    done
}

# full-parallel: one run_full_experiments.sh per partition, all at once.
# Every run keeps its seed, partition and schedule, so results equal the
# sequential ones; only the combined workbook is merged once at the end.
job_parallel() {
    local root="${RESULTS_ROOT:-results/full_experiment}"
    local count; count=$(echo $PARALLEL_TASKS | wc -w)
    local cores; cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    local threads="${THREADS_PER_RUN:-$(( cores / count > 1 ? cores / count : 1 ))}"
    echo "===== $(date '+%F %T') dependency and data check ====="
    bash run.sh --help > /dev/null || { echo "===== dependency setup failed ====="; return 1; }
    bash fetch_data.sh || { echo "===== data preparation failed ====="; return 1; }
    local pids=() task plog fail=0
    for task in $PARALLEL_TASKS; do
        plog="${LOG%.log}_$(partition_dir "$task").log"
        echo "$plog" >> "$CURRENT"
        echo "===== $(date '+%F %T') start $task ($threads CPU threads) -> $plog ====="
        TASKS="$task" PER_PARTITION_SUMMARY=1 SKIP_FINAL_SUMMARY=1 \
            bash run_full_experiments.sh ${EXTRA[@]+"${EXTRA[@]}"} --torch-threads "$threads" \
            > "$plog" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    echo "===== $(date '+%F %T') partitions finished (failures: $fail); merging ====="
    PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" \
        "${PYTHON_BIN:-python3}" summarize.py --results-root "$root" || fail=1
    [ "$fail" = 0 ] || echo "===== some partition failed; check its log, then rerun to resume ====="
    return "$fail"
}

if [ "${SEG_DETACHED:-0}" = "1" ]; then   # the background copy of this script
    echo $$ > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    if [ "$MODE" = "full-parallel" ]; then job_parallel; else job; fi
    exit $?
fi
echo "$LOG" > "$CURRENT"

if [ "${FOREGROUND:-0}" = "1" ]; then
    if [ "$MODE" = "full-parallel" ]; then job_parallel 2>&1 | tee "$LOG"; else job 2>&1 | tee "$LOG"; fi
    exit "${PIPESTATUS[0]}"
fi

# nohup first so SIGHUP is ignored before anything else happens, then setsid
# to leave the terminal / Jupyter kernel session; the run survives both.
LAUNCH=(nohup)
if command -v setsid >/dev/null 2>&1; then LAUNCH=(nohup setsid); fi
SEG_DETACHED=1 "${LAUNCH[@]}" bash "$PROJECT_DIR/start.sh" "$MODE" > "$LOG" 2>&1 < /dev/null &
sleep 5
if running; then
    echo "started $MODE (pid $(cat "$PID_FILE"), device $FORCE_DEVICE)"
    echo "log:    tail -f $LOG"
    if [ "$MODE" = "full-parallel" ]; then
        echo "        one log per partition: ${LOG%.log}_<partition>.log"
    fi
    echo "status: bash start.sh status    stop: bash start.sh stop"
else
    echo "the run stopped right after starting; log follows:" >&2
    cat "$LOG" >&2
    exit 1
fi
