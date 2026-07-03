#!/bin/sh
# Daily research pipeline for the alpha system.
#
# Usage:
#   ./run.sh                  daily: refresh data + features + regime,
#                             retrain live boosters, produce today's ranking.
#                             Auto-refreshes EDGAR fundamentals when the
#                             bulk file is >25 days old, and auto-reruns the
#                             full walk-forward backtest when predictions
#                             are >28 days old (the spec's monthly loop).
#   ./run.sh --sync           also run yahoostock sync_yahoo.py first (updates the DB)
#   ./run.sh --fundamentals   force EDGAR companyfacts re-download + rebuild
#   ./run.sh --with-backtest  force the full walk-forward backtest (~1h)
#   ./run.sh --trade          after ranking, run the alpha bot with --live-trade
#                             (routes the weekly target to the tradingview-mcp
#                             Alpaca service; paper unless that service is LIVE)
set -e
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG"

SYNC=0
BACKTEST=0
FUNDAMENTALS=0
TRADE=0
for arg in "$@"; do
    case "$arg" in
        --sync) SYNC=1 ;;
        --with-backtest) BACKTEST=1 ;;
        --fundamentals) FUNDAMENTALS=1 ;;
        --trade) TRADE=1 ;;
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

# --- Fundamentals: monthly EDGAR refresh (quarterly filings drive it) ---
FACTS=data/companyfacts.zip
if [ "$FUNDAMENTALS" = 1 ] || [ ! -f "$FACTS" ] || [ -n "$(find "$FACTS" -mtime +25 2>/dev/null)" ]; then
    echo "== refreshing EDGAR companyfacts (bulk ~1.3GB) =="
    curl -sL --user-agent "gangwu research wugangc@gmail.com" \
        -o "$FACTS.tmp" \
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip" \
        && mv "$FACTS.tmp" "$FACTS"
    run python3 scripts/08_fundamentals.py
else
    echo "== EDGAR fundamentals fresh (<25 days) — skipping refresh =="
fi

run python3 scripts/02_features.py 6
run python3 scripts/refresh_regime.py
run python3 scripts/07_train_latest.py

# --- Monthly walk-forward retrain + full backtest + report refresh ---
PREDS=data/predictions_20d.parquet
if [ "$BACKTEST" = 1 ] || [ ! -f "$PREDS" ] || [ -n "$(find "$PREDS" -mtime +28 2>/dev/null)" ]; then
    run python3 scripts/03_backtest.py
    run python3 scripts/05_export_report.py
else
    echo "== walk-forward predictions fresh (<28 days) — skipping full backtest =="
fi

run python3 scripts/04_rank_today.py

if [ "$TRADE" = 1 ]; then
    run python3 scripts/09_alpha_bot.py --live-trade
fi

# --- Housekeeping: prune run logs and daily rank CSVs older than 90 days ---
find logs -name 'run_*.log' -mtime +90 -delete 2>/dev/null || true
find reports -name 'rank_*.csv' -mtime +90 -delete 2>/dev/null || true

echo "pipeline complete — reports/ has the latest ranking; dashboard reads the new artifacts automatically"
