#!/usr/bin/env python3
"""
rsi_pivot_bot.py — paper-trading bot for the weekly RSI-pivot strategy.

Locked to the best backtested config (5y S&P 500):
    EMA200 trend filter + 20 slots + SPY-overlay on idle cash
    -> +84.6% total / 13.1% CAGR / Sharpe 0.83 (edges SPY B&H).

Rules (identical to backtest_rsi_pivot / scan_rsi_pivot):
    rsiFast = SMA( RSI(close, 7), 5 )
    BUY  = fast-RSI pivot valley (turned up) AND fast RSI < 50 AND close > EMA200
    SELL = fast RSI crosses DOWN through 50
    - Up to 20 concurrent positions, equal-weight of current equity.
    - Deepest dip (lowest fast RSI) fills first when slots are scarce.
    - Idle cash is modeled as parked in SPY (overlay) for equity accounting.
    - No stop (stops hurt this strategy in every test).

This is a PAPER bot: it keeps a JSON state file (positions/cash/equity), and on
each run emits the BUY/SELL orders for the just-closed weekly bar. It never places
real orders. Run it once per week after the Friday close (or with --dry-run anytime).

Usage:
    python3 rsi_pivot_bot.py --file sp500.txt                 # run against a universe
    python3 rsi_pivot_bot.py --file sp500.txt --dry-run       # show orders, don't persist
    python3 rsi_pivot_bot.py --file sp500.txt --state my.json # custom state file
    python3 rsi_pivot_bot.py --status                         # print current paper book
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_rsi_pivot import fast_rsi, load_tickers
from alpaca_client import TradingClient

# ---- LOCKED best-backtest parameters ---------------------------------------
FAST, SMOOTH = 7, 5
MID = 50.0
TREND_EMA = 200          # only buy dips above EMA200
SLOTS = 20               # max concurrent positions
SPY_OVERLAY = True       # idle cash parked in SPY
START_CASH = 100_000.0
DEFAULT_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "report", "rsi_pivot_bot_state.json")


def compute_signals(close: pd.Series):
    """Return (buy, sell, rf) boolean Series confirmed on each CLOSED bar."""
    rf = fast_rsi(close, FAST, SMOOTH)
    rf1, rf2 = rf.shift(1), rf.shift(2)
    buy = (rf > rf1) & (rf1 <= rf2) & (rf < MID)
    sell = (rf < MID) & (rf1 >= MID)
    ema = close.ewm(span=TREND_EMA, adjust=False, min_periods=TREND_EMA).mean()
    buy = buy & (close > ema)
    return buy.fillna(False), sell.fillna(False), rf


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": START_CASH, "positions": {}, "history": [], "last_bar": None}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def print_status(state, prices=None):
    pos = state["positions"]
    print(f"\n=== Paper book ===")
    print(f"Cash (idle, SPY-overlay): ${state['cash']:,.0f}")
    print(f"Open positions: {len(pos)}/{SLOTS}")
    if pos:
        eq = state["cash"]
        print(f"{'sym':<8}{'shares':>10}{'entry':>10}{'last':>10}{'value':>12}{'P&L%':>8}")
        for sym, p in sorted(pos.items()):
            last = prices.get(sym) if prices else p["entry"]
            val = p["shares"] * last
            pnl = (last / p["entry"] - 1) * 100
            eq += val
            print(f"{sym:<8}{p['shares']:>10.2f}{p['entry']:>10.2f}{last:>10.2f}{val:>12,.0f}{pnl:>+8.1f}")
        print(f"{'TOTAL EQUITY':<8}{'':>32}{eq:>12,.0f}")
    else:
        print("(flat — all in SPY overlay)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="universe ticker file")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--state", default=DEFAULT_STATE, help="paper-book state JSON")
    ap.add_argument("--dry-run", action="store_true", help="emit orders, do not persist")
    ap.add_argument("--status", action="store_true", help="print current paper book and exit")
    ap.add_argument("--live-trade", action="store_true",
                    help="send each BUY/SELL to trading_service (Alpaca paper by default)")
    args = ap.parse_args()

    state = load_state(args.state)

    if args.status and not (args.file or args.tickers):
        print_status(state)
        return

    tickers = load_tickers(args)
    if not tickers:
        ap.error("no universe given (pass --file or symbols)")

    # fetch enough history for EMA200 weekly (+ SPY for overlay & prices)
    data = yf.download(tickers + ["SPY"], interval="1wk", period="6y",
                       group_by="ticker", auto_adjust=True, progress=False, threads=True)

    # determine the latest CLOSED weekly bar shared across the data
    spy_close = data["SPY"]["Close"].dropna()
    latest_bar = spy_close.index[-1]

    # gather signals + latest prices for the current bar
    buys, sells, last_px = [], [], {}
    for sym in tickers:
        try:
            close = data[sym]["Close"].dropna()
            if len(close) < TREND_EMA + 5:
                continue
            close = close[close.index <= latest_bar]
            buy, sell, rf = compute_signals(close)
            last_px[sym] = float(close.iloc[-1])
            if bool(buy.iloc[-1]):
                buys.append((sym, float(rf.iloc[-1])))
            if bool(sell.iloc[-1]):
                sells.append((sym, float(rf.iloc[-1])))
        except Exception:  # noqa: BLE001
            continue
    last_px["SPY"] = float(spy_close.iloc[-1])

    # SPY overlay: grow idle cash by SPY's return since the last processed bar
    prev_bar = state.get("last_bar")
    if SPY_OVERLAY and prev_bar and str(prev_bar) != str(latest_bar):
        try:
            prev_spy = float(spy_close[spy_close.index <= pd.Timestamp(prev_bar)].iloc[-1])
            state["cash"] *= last_px["SPY"] / prev_spy
        except Exception:  # noqa: BLE001
            pass

    pos = state["positions"]
    orders = {"sell": [], "buy": []}

    # 1) SELLS — close any held position that fired a SELL
    for sym, rf in sells:
        if sym in pos:
            px = last_px.get(sym, pos[sym]["entry"])
            proceeds = pos[sym]["shares"] * px
            state["cash"] += proceeds
            orders["sell"].append({"sym": sym, "px": round(px, 2),
                                   "ret_pct": round((px / pos[sym]["entry"] - 1) * 100, 1)})
            del pos[sym]

    # 2) BUYS — fill free slots, deepest dip (lowest fast RSI) first
    buys = [b for b in buys if b[0] not in pos]
    buys.sort(key=lambda b: b[1])
    free = SLOTS - len(pos)
    # current equity = cash + held mark-to-market
    equity = state["cash"] + sum(p["shares"] * last_px.get(s, p["entry"]) for s, p in pos.items())
    for sym, rf in buys[:max(0, free)]:
        alloc = min(equity / SLOTS, state["cash"])
        if alloc <= 0:
            break
        px = last_px[sym]
        shares = alloc / px
        state["cash"] -= shares * px
        pos[sym] = {"shares": round(shares, 4), "entry": round(px, 2),
                    "entry_bar": str(latest_bar.date())}
        orders["buy"].append({"sym": sym, "px": round(px, 2), "fastRSI": round(rf, 1)})

    # report
    print(f"\n=== RSI-Pivot bot | bar {latest_bar.date()} | EMA200+20slot+SPYoverlay ===")
    if str(prev_bar) == str(latest_bar):
        print("(no new weekly bar since last run — orders below are for the same bar)")
    print(f"\nSELL ({len(orders['sell'])}): " +
          (", ".join(f"{o['sym']}@{o['px']}({o['ret_pct']:+}%)" for o in orders["sell"]) or "none"))
    print(f"BUY  ({len(orders['buy'])}): " +
          (", ".join(f"{o['sym']}@{o['px']}(RSI{o['fastRSI']})" for o in orders["buy"]) or "none"))

    # route orders to the trading service (Alpaca paper) if requested
    if args.live_trade and not args.dry_run:
        tc = TradingClient()
        if not tc.enabled:
            print("live-trade: TRADING_SERVICE_URL not set — skipping broker orders")
        else:
            equity = state["cash"] + sum(p["shares"] * last_px.get(s, p["entry"])
                                         for s, p in pos.items())
            slot_notional = round(equity / SLOTS, 2)
            for o in orders["sell"]:
                print(f"live-trade SELL {o['sym']}: {tc.liquidate(o['sym'])}")
            for o in orders["buy"]:
                print(f"live-trade BUY {o['sym']} ${slot_notional}: "
                      f"{tc.order(o['sym'], 'buy', notional=slot_notional)}")

    print_status(state, last_px)

    if not args.dry_run:
        state["last_bar"] = str(latest_bar.date())
        state["history"].append({"bar": str(latest_bar.date()),
                                 "sells": orders["sell"], "buys": orders["buy"],
                                 "cash": round(state["cash"], 2)})
        save_state(args.state, state)
        print(f"\nState saved -> {args.state}")
    else:
        print("\n(dry-run — state NOT saved)")


if __name__ == "__main__":
    main()
