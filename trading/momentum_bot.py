#!/usr/bin/env python3
"""
momentum_bot.py — paper bot for the cross-sectional MOMENTUM strategy.

Rule (rebalance every 4 weeks):
    Hold the top-N S&P 500 names by 26-week (6-month) price momentum,
    equal weight. Rebalance monthly; between rebalances, hold.

Backtest (8y, no lookahead, CURRENT S&P list): +2822% / 52.5% CAGR / Sharpe 1.51,
vs SPY +205% / 15.0% / 0.85 — BUT that magnitude is inflated by SURVIVORSHIP BIAS
(applying today's index members to the past) and ignores costs/turnover. The live
signal (ranking today's real prices) is legitimate; temper return expectations.

Paper only: JSON state, --dry-run/--status, idempotent within a rebalance cycle.
    python3 momentum_bot.py --file sp500.txt
    python3 momentum_bot.py --file sp500.txt --dry-run
    python3 momentum_bot.py --status
    python3 momentum_bot.py --file sp500.txt --topn 15 --rebalance 4
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_rsi_pivot import load_tickers
from alpaca_client import TradingClient

MOM_WK = 26
TOPN = 15
REBAL_WK = 4
START_CASH = 100_000.0
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DEFAULT_STATE = os.path.join(STATE_DIR, "momentum_bot_state.json")


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": START_CASH, "positions": {}, "last_bar": None,
            "last_rebalance": None, "history": []}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def weeks_between(idx, d0, d1):
    """Number of weekly bars in idx strictly between d0 and d1 (inclusive of d1)."""
    if d0 is None:
        return REBAL_WK  # force rebalance on first run
    mask = (idx > pd.Timestamp(d0)) & (idx <= pd.Timestamp(d1))
    return int(mask.sum())


def print_status(state, prices=None):
    pos = state["positions"]
    print(f"\n=== MOMENTUM paper book ===")
    print(f"Cash: ${state['cash']:,.0f}   holdings: {len(pos)}")
    print(f"last_bar: {state.get('last_bar')}  last_rebalance: {state.get('last_rebalance')}")
    if pos:
        eq = state["cash"]
        print(f"{'sym':<8}{'shares':>10}{'entry':>10}{'last':>10}{'value':>12}{'P&L%':>8}")
        for s, p in sorted(pos.items()):
            last = prices.get(s) if prices else p["entry"]
            val = p["shares"] * last
            eq += val
            pnl = (last / p["entry"] - 1) * 100
            print(f"{s:<8}{p['shares']:>10.2f}{p['entry']:>10.2f}{last:>10.2f}{val:>12,.0f}{pnl:>+8.1f}")
        print(f"{'EQUITY':<8}{'':>32}{eq:>12,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--topn", type=int, default=TOPN)
    ap.add_argument("--rebalance", type=int, default=REBAL_WK, help="weeks between rebalances")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--live-trade", action="store_true",
                    help="on rebalance, send target weights to trading_service (Alpaca paper)")
    args = ap.parse_args()

    state = load_state(args.state)

    if args.status and not (args.file or args.tickers):
        print_status(state)
        return

    universe = load_tickers(args)
    if not universe:
        ap.error("no universe (pass --file or symbols)")

    data = yf.download(universe, interval="1wk", period="2y", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)

    # build a close matrix, compute 26wk momentum on the latest bar
    closes = {}
    for s in universe:
        try:
            c = data[s]["Close"].dropna()
            if len(c) > MOM_WK + 2:
                closes[s] = c
        except Exception:  # noqa: BLE001
            continue
    px = pd.DataFrame(closes).dropna(how="all")
    bar = px.index[-1]
    last_px = {s: float(px[s].iloc[-1]) for s in px.columns if not pd.isna(px[s].iloc[-1])}
    mom = (px.iloc[-1] / px.iloc[-1 - MOM_WK] - 1.0).dropna()

    # is it time to rebalance?
    n_since = weeks_between(px.index, state.get("last_rebalance"), bar)
    do_rebal = n_since >= args.rebalance
    already = str(state.get("last_bar")) == str(bar.date())

    print(f"\n=== MOMENTUM bot | bar {bar.date()} | top{args.topn} by {MOM_WK}wk mom ===")
    if already:
        print("(no new weekly bar since last run)")

    orders = {"sell": [], "buy": []}
    pos = state["positions"]

    if do_rebal and not already:
        target = list(mom.nlargest(args.topn).index)
        # SELL names no longer in target
        for s in list(pos.keys()):
            if s not in target and s in last_px:
                state["cash"] += pos[s]["shares"] * last_px[s]
                orders["sell"].append(s)
                del pos[s]
        # equity after sells
        equity = state["cash"] + sum(p["shares"] * last_px.get(s, p["entry"]) for s, p in pos.items())
        target_val = equity / args.topn
        # BUY new names to equal weight (only add missing ones; keep existing)
        for s in target:
            if s not in pos and s in last_px:
                alloc = min(target_val, state["cash"])
                if alloc <= 0:
                    break
                shares = alloc / last_px[s]
                state["cash"] -= shares * last_px[s]
                pos[s] = {"shares": round(shares, 4), "entry": round(last_px[s], 2),
                          "mom": round(float(mom[s]) * 100, 1)}
                orders["buy"].append(s)
        state["last_rebalance"] = str(bar.date())
        print(f"REBALANCE — target top{args.topn}: {', '.join(target)}")
        print(f"SELL ({len(orders['sell'])}): {', '.join(orders['sell']) or 'none'}")
        print(f"BUY  ({len(orders['buy'])}): {', '.join(orders['buy']) or 'none'}")
        # route equal-weight target to the trading service (Alpaca paper)
        if args.live_trade and not args.dry_run:
            tc = TradingClient()
            if not tc.enabled:
                print("live-trade: TRADING_SERVICE_URL not set — skipping broker rebalance")
            else:
                w = round(1.0 / args.topn, 4)
                res = tc.rebalance({s: w for s in target})
                print(f"live-trade rebalance: {res}")
    else:
        nxt = args.rebalance - n_since
        print(f"HOLD — next rebalance in ~{max(0,nxt)} week(s). Current top{args.topn} preview: "
              f"{', '.join(mom.nlargest(args.topn).index)}")

    print_status(state, last_px)

    if not args.dry_run:
        state["last_bar"] = str(bar.date())
        state["history"].append({"bar": str(bar.date()), "rebalanced": do_rebal and not already,
                                 "sells": orders["sell"], "buys": orders["buy"],
                                 "cash": round(state["cash"], 2)})
        save_state(args.state, state)
        print(f"\nState saved -> {args.state}")
    else:
        print("\n(dry-run — state NOT saved)")


if __name__ == "__main__":
    main()
