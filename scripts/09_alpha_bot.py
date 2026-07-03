#!/usr/bin/env python3
"""
09_alpha_bot.py — execution bot for the alpha ranking system, in the same
mold as tradingview-mcp's momentum/lev_trend/rsi_pivot bots.

Signal source: reports/target_portfolio.json (written by 04_rank_today,
turnover-aware top-20 with regime-scaled gross). The bot never computes
signals — it publishes the pipeline's target to a paper book and,
with --live-trade, to the Alpaca trading service in tradingview-mcp
(PAPER unless that service was started with ALPACA_LIVE=1; this bot
additionally refuses a LIVE service without --live-ok).

    python3 scripts/09_alpha_bot.py --dry-run     # show what would trade
    python3 scripts/09_alpha_bot.py               # update local paper book
    python3 scripts/09_alpha_bot.py --live-trade  # also route to Alpaca service
    python3 scripts/09_alpha_bot.py --status      # print the paper book

Cadence: weekly (ISO week of the target's as-of date), matching the
backtest's W-MON rebalance; idempotent within a week. --force overrides.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = Path(__file__).resolve().parent.parent
TVMCP = ALPHA / "trading"
sys.path.insert(0, str(ALPHA))
sys.path.insert(0, str(TVMCP))

TARGET_PATH = ALPHA / "reports" / "target_portfolio.json"
STATE_PATH = ALPHA / "reports" / "alpha_bot_state.json"
TRADES_PATH = ALPHA / "reports" / "alpha_bot_trades.csv"
START_CASH = 100_000.0
MAX_STALE_BDAYS = 3
MAX_POSITION = 0.11        # sanity ceilings on the published target
MAX_GROSS = 1.30
MIN_NAMES = 5


def _load_service_env():
    """Same env the other bots get from run_bot.sh: service URL + token."""
    env_file = TVMCP / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault("TRADING_SERVICE_URL", "http://127.0.0.1:8787")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"cash": START_CASH, "positions": {}, "last_rebalance": None,
            "last_asof": None, "history": []}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def book_equity(state, px):
    return state["cash"] + sum(p["shares"] * px.get(s, p["entry"])
                               for s, p in state["positions"].items())


def print_status(state, px=None):
    px = px or {}
    pos = state["positions"]
    print("\n=== ALPHA paper book ===")
    print(f"Cash: ${state['cash']:,.0f}   holdings: {len(pos)}   "
          f"last_rebalance: {state.get('last_rebalance')}")
    if pos:
        eq = state["cash"]
        print(f"{'sym':<8}{'shares':>10}{'entry':>10}{'last':>10}{'value':>12}{'P&L%':>8}")
        for s, p in sorted(pos.items()):
            last = px.get(s, p["entry"])
            val = p["shares"] * last
            eq += val
            pnl = (last / p["entry"] - 1) * 100
            print(f"{s:<8}{p['shares']:>10.2f}{p['entry']:>10.2f}{last:>10.2f}"
                  f"{val:>12,.0f}{pnl:>+8.1f}")
        print(f"{'EQUITY':<8}{'':>32}{eq:>12,.0f}")


def validate_target(t):
    asof = pd.Timestamp(t["asof"])
    stale = int(np.busday_count(asof.date(), pd.Timestamp.now().date()))
    if stale > MAX_STALE_BDAYS:
        sys.exit(f"REFUSED: target is {stale} business days old (asof {t['asof']}) "
                 f"— run the pipeline first")
    w = t["weights"]
    if len(w) < MIN_NAMES:
        sys.exit(f"REFUSED: only {len(w)} names in target")
    if max(w.values()) > MAX_POSITION:
        sys.exit(f"REFUSED: position {max(w, key=w.get)} weight "
                 f"{max(w.values()):.1%} > {MAX_POSITION:.0%} ceiling")
    if sum(w.values()) > MAX_GROSS:
        sys.exit(f"REFUSED: gross {sum(w.values()):.2f} > {MAX_GROSS} ceiling")
    return asof


def latest_prices(symbols):
    """Close prices from the pipeline's own snapshot (adjusted, as-of run)."""
    px = pd.read_parquet(ALPHA / "data" / "prices.parquet",
                         columns=["ticker", "date", "close"])
    px = px[px["ticker"].isin({s + ".US" for s in symbols})]
    last = px.sort_values("date").groupby("ticker")["close"].last()
    return {t.replace(".US", ""): float(v) for t, v in last.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--live-trade", action="store_true",
                    help="route the rebalance to the tradingview-mcp trading service")
    ap.add_argument("--live-ok", action="store_true",
                    help="required extra ack if the service is in LIVE (real money) mode")
    ap.add_argument("--force", action="store_true", help="ignore the weekly cadence guard")
    ap.add_argument("--cash-buffer", type=float, default=0.02)
    args = ap.parse_args()

    state = load_state()
    if args.status:
        pos_syms = list(state["positions"])
        print_status(state, latest_prices(pos_syms) if pos_syms else {})
        return

    if not TARGET_PATH.exists():
        sys.exit("no reports/target_portfolio.json — run scripts/04_rank_today.py first")
    target = json.loads(TARGET_PATH.read_text())
    asof = validate_target(target)
    weights = {s: float(w) for s, w in target["weights"].items()}

    # weekly cadence, aligned with the backtest's W-MON rebalance
    this_week = f"{asof.isocalendar().year}-W{asof.isocalendar().week:02d}"
    if state.get("last_rebalance") == this_week and not args.force:
        print(f"HOLD — already rebalanced in {this_week} "
              f"(asof {target['asof']}, regime {target['regime']}). --force to override.")
        return

    px = latest_prices(set(weights) | set(state["positions"]))
    equity = book_equity(state, px)
    held = set(state["positions"])
    sells = sorted(held - set(weights))
    buys = sorted(set(weights) - held)

    print(f"=== ALPHA bot | asof {target['asof']} | regime {target['regime']} "
          f"| gross {target['gross']:.0%} | {len(weights)} names ===")
    print(f"paper equity: ${equity:,.0f}")
    print(f"SELL ({len(sells)}): {', '.join(sells) or 'none'}")
    print(f"BUY  ({len(buys)}): {', '.join(buys) or 'none'}")
    print(f"target: " + ", ".join(f"{s}:{w:.1%}" for s, w in
                                  sorted(weights.items(), key=lambda x: -x[1])))
    if args.dry_run:
        print("(dry run — no state or orders changed)")
        return

    # --- update the local paper book to the target ---
    state["positions"] = {
        s: {"shares": round(equity * w / px[s], 4), "entry": round(px[s], 2),
            "weight": w}
        for s, w in weights.items() if s in px and px[s] > 0
    }
    invested = sum(p["shares"] * p["entry"] for p in state["positions"].values())
    state["cash"] = round(equity - invested, 2)
    state["last_rebalance"] = this_week
    state["last_asof"] = target["asof"]
    state["history"].append({"week": this_week, "asof": target["asof"],
                             "equity": round(equity, 2), "n": len(weights),
                             "regime": target["regime"]})
    save_state(state)
    print(f"paper book updated ({len(state['positions'])} positions, "
          f"cash ${state['cash']:,.0f})")

    # --- route to the Alpaca trading service ---
    mode = None
    if args.live_trade:
        _load_service_env()
        from alpaca_client import TradingClient
        tc = TradingClient()
        health = tc.health() or {}
        mode = health.get("mode")
        if health.get("status") != 200:
            print(f"live-trade SKIPPED: trading service unreachable ({health})")
        elif mode == "LIVE" and not args.live_ok:
            print("live-trade REFUSED: service is in LIVE (real money) mode — "
                  "pass --live-ok to acknowledge")
        else:
            res = tc.rebalance(weights, cash_buffer=args.cash_buffer)
            print(f"live-trade rebalance [{mode}]: {res}")

    row = pd.DataFrame([{
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "week": this_week, "asof": target["asof"], "regime": target["regime"],
        "gross": target["gross"], "n_names": len(weights),
        "sells": len(sells), "buys": len(buys),
        "paper_equity": round(equity, 2),
        "routed": bool(args.live_trade), "service_mode": mode,
    }])
    row.to_csv(TRADES_PATH, mode="a", header=not TRADES_PATH.exists(), index=False)


if __name__ == "__main__":
    main()
