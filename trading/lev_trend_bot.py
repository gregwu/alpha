#!/usr/bin/env python3
"""
lev_trend_bot.py — paper bot for the LEV-TREND strategy.

Rule (weekly):
    Hold a leveraged S&P proxy (default SSO 2x; UPRO 3x optional) while
    SPY closes ABOVE its 40-week (~200-day) SMA; otherwise hold cash.
    Decision uses the just-closed weekly bar; fill at that close (paper).

Backtest (8y, no lookahead): +416% total / 22.8% CAGR / Sharpe 0.98 / DD -28%,
vs SPY +205% / 15.0% / 0.85. The 200d filter cut the 2022 loss to ~SPY despite
2x leverage. No survivorship bias (single ETF + filter).

Paper only: keeps a JSON state file, prints the target position each run.
    python3 lev_trend_bot.py               # weekly run (persists)
    python3 lev_trend_bot.py --dry-run
    python3 lev_trend_bot.py --status
    python3 lev_trend_bot.py --etf UPRO     # use 3x instead of 2x
"""
import argparse
import json
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpaca_client import TradingClient

SMA_WK = 40
DEFAULT_ETF = "SSO"       # 2x S&P 500; use UPRO for 3x
START_CASH = 100_000.0
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
DEFAULT_STATE = os.path.join(STATE_DIR, "lev_trend_bot_state.json")


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": START_CASH, "holding": None, "shares": 0.0,
            "entry": None, "last_bar": None, "history": []}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etf", default=DEFAULT_ETF, help="leveraged ETF to hold (SSO=2x, UPRO=3x)")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--live-trade", action="store_true",
                    help="also send orders to trading_service (Alpaca paper by default)")
    args = ap.parse_args()

    state = load_state(args.state)

    if args.status:
        h = state.get("holding")
        print(f"\n=== LEV-TREND paper book ===")
        print(f"Holding: {h or 'CASH'}")
        if h:
            print(f"  shares={state['shares']:.2f} entry={state['entry']}")
        print(f"Cash: ${state['cash']:,.0f}   last_bar: {state.get('last_bar')}")
        return

    data = yf.download(["SPY", args.etf], interval="1wk", period="3y",
                       group_by="ticker", auto_adjust=True, progress=False, threads=True)
    spy = data["SPY"]["Close"].dropna()
    etf = data[args.etf]["Close"].dropna()
    bar = spy.index[-1]
    sma = spy.rolling(SMA_WK).mean().iloc[-1]
    spy_last = float(spy.iloc[-1])
    etf_last = float(etf.iloc[-1])
    above = spy_last > sma
    target = args.etf if above else "CASH"

    print(f"\n=== LEV-TREND bot | bar {bar.date()} | {args.etf} + SPY>SMA{SMA_WK} ===")
    print(f"SPY {spy_last:.2f} vs SMA{SMA_WK} {sma:.2f}  ->  {'ABOVE (risk-on)' if above else 'BELOW (risk-off)'}")

    cur = state.get("holding")
    order = None
    if target == args.etf and cur != args.etf:
        # buy the ETF with all cash
        shares = state["cash"] / etf_last
        state.update(holding=args.etf, shares=shares, entry=round(etf_last, 2), cash=0.0)
        order = f"BUY {args.etf} {shares:.2f}sh @ {etf_last:.2f}"
    elif target == "CASH" and cur == args.etf:
        proceeds = state["shares"] * etf_last
        ret = (etf_last / state["entry"] - 1) * 100 if state.get("entry") else 0.0
        state.update(holding=None, shares=0.0, entry=None, cash=proceeds)
        order = f"SELL {args.etf} @ {etf_last:.2f} ({ret:+.1f}%) -> CASH"
    else:
        order = f"HOLD {cur or 'CASH'} (no change)"

    # mark-to-market equity
    equity = state["cash"] + (state["shares"] * etf_last if state.get("holding") else 0.0)
    print(f"Order: {order}")
    print(f"Equity: ${equity:,.0f}  (holding {state.get('holding') or 'CASH'})")

    # optionally route the order to the live trading service (Alpaca paper)
    if args.live_trade and not args.dry_run:
        tc = TradingClient()
        if not tc.enabled:
            print("live-trade: TRADING_SERVICE_URL not set — skipping broker order")
        elif order.startswith("BUY"):
            # invest the whole account into the ETF (notional = current buying power)
            acct = tc.account() or {}
            bp = float(acct.get("cash", 0) or 0) * 0.98  # 2% buffer for slippage/reserved cash
            res = tc.order(args.etf, "buy", notional=round(bp, 2)) if bp > 1 else None
            print(f"live-trade: {res}")
        elif order.startswith("SELL"):
            print(f"live-trade: {tc.liquidate(args.etf)}")
        else:
            print("live-trade: no change, no broker order")

    if not args.dry_run:
        state["last_bar"] = str(bar.date())
        state["history"].append({"bar": str(bar.date()), "target": target,
                                 "order": order, "equity": round(equity, 2)})
        save_state(args.state, state)
        print(f"State saved -> {args.state}")
    else:
        print("(dry-run — state NOT saved)")


if __name__ == "__main__":
    main()
