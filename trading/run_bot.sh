#!/usr/bin/env bash
#
# run_bot.sh — convenience wrapper for the trading service + strategy bots.
#
# Loads Alpaca keys and the service token from .env, exports the vars the bots
# need (TRADING_SERVICE_URL / TRADING_SERVICE_TOKEN), and dispatches to a bot or
# the service. Defaults to PAPER trading (the service only goes live if
# ALPACA_LIVE=1 is set in .env).
#
# Usage:
#   ./run_bot.sh service                 # start the trading webservice (foreground)
#   ./run_bot.sh health                  # curl the service /health
#   ./run_bot.sh account                 # curl the service /account
#
#   ./run_bot.sh rsi        [--live-trade] [extra args]   # RSI-pivot bot
#   ./run_bot.sh lev        [--live-trade] [extra args]   # LEV-TREND bot
#   ./run_bot.sh momentum   [--live-trade] [extra args]   # MOMENTUM bot
#   ./run_bot.sh all        [--live-trade]                # run all three bots
#
#   ./run_bot.sh status                  # print every bot's paper book
#
# Without --live-trade, bots run in state-file paper mode (no broker orders).
# With --live-trade, they route orders through the running service.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
PORT="${PORT:-8787}"
UNIVERSE="${UNIVERSE:-$HERE/sp500.txt}"

# ---- load .env (export every KEY=VALUE, ignore comments/blanks) -------------
if [[ -f .env ]]; then
  set -a
  while IFS= read -r line; do
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    eval "$line"
  done < .env
  set +a
else
  echo "run_bot.sh: no .env found in $HERE" >&2
fi

# service coordinates the bots use
export TRADING_SERVICE_URL="${TRADING_SERVICE_URL:-http://127.0.0.1:$PORT}"
export TRADING_SERVICE_TOKEN="${TRADING_SERVICE_TOKEN:-local-dev-token}"

TOKEN_HDR=(-H "X-Service-Token: ${TRADING_SERVICE_TOKEN}")

cmd="${1:-}"; shift || true

case "$cmd" in
  service)
    echo "Starting trading service on port $PORT (mode: ${ALPACA_LIVE:+LIVE}${ALPACA_LIVE:-PAPER})..."
    exec "$PY" trading_service.py
    ;;
  health)
    curl -s "${TOKEN_HDR[@]}" "$TRADING_SERVICE_URL/health"; echo
    ;;
  account)
    curl -s "${TOKEN_HDR[@]}" "$TRADING_SERVICE_URL/account"; echo
    ;;
  positions)
    curl -s "${TOKEN_HDR[@]}" "$TRADING_SERVICE_URL/positions"; echo
    ;;
  rsi)
    exec "$PY" rsi_pivot_bot.py --file "$UNIVERSE" "$@"
    ;;
  lev)
    exec "$PY" lev_trend_bot.py "$@"
    ;;
  momentum|mom)
    exec "$PY" momentum_bot.py --file "$UNIVERSE" "$@"
    ;;
  all)
    echo "===== LEV-TREND ====="; "$PY" lev_trend_bot.py "$@" || true
    echo "===== MOMENTUM =====";  "$PY" momentum_bot.py --file "$UNIVERSE" "$@" || true
    echo "===== RSI-PIVOT ====="; "$PY" rsi_pivot_bot.py --file "$UNIVERSE" "$@" || true
    ;;
  status)
    echo "===== LEV-TREND ====="; "$PY" lev_trend_bot.py --status || true
    echo "===== MOMENTUM =====";  "$PY" momentum_bot.py --status || true
    echo "===== RSI-PIVOT ====="; "$PY" rsi_pivot_bot.py --status || true
    ;;
  ""|-h|--help|help)
    sed -n '2,30p' "$0"
    ;;
  *)
    echo "run_bot.sh: unknown command '$cmd' (try: service|rsi|lev|momentum|all|status|health)" >&2
    exit 1
    ;;
esac
