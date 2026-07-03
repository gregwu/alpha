#!/bin/sh
# Daily research pipeline for the alpha system.
#
# Usage:
#   ./run.sh                  refresh data + features + regime + live ranking
#   ./run.sh --sync           also run yahoostock sync_yahoo.py first (updates the DB)
#   ./run.sh --with-backtest  additionally re-run the full walk-forward backtest
#                             (monthly retraining, ~20 min) and report export
set -e
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG"

SYNC=0
BACKTEST=0
for arg in "$@"; do
    case "$arg" in
        --sync) SYNC=1 ;;
        --with-backtest) BACKTEST=1 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

run() {
    echo "== $* =="
    "$@" 2>&1 | tee -a "$LOG"
}

if [ "$SYNC" = 1 ]; then
    run sh -c 'cd /Users/gangwu/git/portal/yahoostock && python3 sync_yahoo.py --master-source alpaca'
fi

run python3 scripts/01_snapshot.py
run python3 scripts/02_features.py 6
run python3 scripts/refresh_regime.py

if [ "$BACKTEST" = 1 ]; then
    run python3 scripts/03_backtest.py
    run python3 scripts/05_export_report.py
fi

run python3 scripts/04_rank_today.py
echo "pipeline complete — reports/ has the latest ranking"
