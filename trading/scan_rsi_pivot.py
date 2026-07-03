#!/usr/bin/env python3
"""
scan_rsi_pivot.py — replicate the "RSI Delta Filter" early buy/sell dot in Python.

Signal definition (Fast-RSI-pivot trigger + 50-line zone filter — the backtested
rule; PF 2.37 over 5y S&P 500 vs 1.38 for the old OB/OS(72/28) zone):

  rsiFast = SMA( RSI(close, fastLen), smooth )         # smoothed fast RSI

  BUY  present on current bar when the smoothed fast RSI makes a LOCAL VALLEY that
       turns up on the current (live) bar, and fast RSI < 50:
           rsiFast[t] > rsiFast[t-1] and rsiFast[t-1] <= rsiFast[t-2] and rsiFast[t] < 50

  SELL present on current bar when it makes a LOCAL PEAK that turns down on the
       current bar, and fast RSI > 50:
           rsiFast[t] < rsiFast[t-1] and rsiFast[t-1] >= rsiFast[t-2] and rsiFast[t] > 50

"current bar" = the most recent bar returned by yfinance (the still-forming weekly bar).
This is the same "fake next bar @ close" early trigger you see live on the chart:
the pivot is called on the live bar instead of waiting for the next bar to confirm.

Usage:
  python3 scan_rsi_pivot.py AAPL MSFT NVDA
  python3 scan_rsi_pivot.py --file tickers.txt
  python3 scan_rsi_pivot.py --file tickers.txt --interval 1wk --fast 7 --smooth 5 --mid 50
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf


# ---- indicator params. RSI is fast=7 / smooth=5 (from the live chart study).
#      Zone filter uses the 50 MID line (the backtested rule): BUY only when fast
#      RSI is below 50, SELL only when above 50. This beat the OB/OS(72/28) zone
#      in the 5y S&P 500 backtest (PF 2.37 vs 1.38). ---------------------------
FAST_LEN = 7
SMOOTH = 5
MID_LEVEL = 50.0   # buys only below this, sells only above this


def rsi(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RSI — matches ta.rsi()."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/length
    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_dn = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_dn
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(roll_dn != 0, 100.0)   # all-gains -> 100
    return out


def fast_rsi(close: pd.Series, fast_len: int, smooth: int) -> pd.Series:
    """SMA(RSI(close, fast_len), smooth) — the smoothed fast RSI line."""
    return rsi(close, fast_len).rolling(smooth).mean()


def ema200_state(close: pd.Series, length: int = 200):
    """
    Return (ema_value, below) for the current bar, or (None, None) if there is
    not enough history for a full-length EMA. 'below' is True when close < EMA200.
    """
    if len(close) < length:
        return None, None
    ema = close.ewm(span=length, adjust=False, min_periods=length).mean()
    ev = ema.iloc[-1]
    if pd.isna(ev):
        return None, None
    return float(ev), bool(close.iloc[-1] < ev)


def evaluate(close: pd.Series, fast_len=FAST_LEN, smooth=SMOOTH, mid=MID_LEVEL):
    """
    Return (signal, rf_now, rf1, rf2) for the CURRENT (last) bar.
    signal is 'BUY', 'SELL', or None.

    Pivot logic with a 50-line zone filter:
      BUY  = fast-RSI valley (turned up)  AND fast RSI < 50
      SELL = fast-RSI peak   (turned down) AND fast RSI > 50
    """
    rf = fast_rsi(close, fast_len, smooth).dropna()
    if len(rf) < 3:
        return None, None, None, None
    rf_now, rf1, rf2 = rf.iloc[-1], rf.iloc[-2], rf.iloc[-3]

    buy = (rf_now > rf1) and (rf1 <= rf2) and (rf_now < mid)     # valley turns up, below 50
    sell = (rf_now < rf1) and (rf1 >= rf2) and (rf_now > mid)    # peak turns down, above 50

    sig = "BUY" if buy else "SELL" if sell else None
    return sig, float(rf_now), float(rf1), float(rf2)


def load_tickers(args) -> list[str]:
    syms: list[str] = list(args.tickers)
    if args.file:
        with open(args.file) as f:
            for line in f:
                for tok in line.replace(",", " ").split():
                    tok = tok.strip().upper()
                    if tok and not tok.startswith("#"):
                        syms.append(tok)
    # de-dup, preserve order
    seen, out = set(), []
    for s in syms:
        s = s.upper()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Scan tickers for RSI-pivot buy/sell on the current bar.")
    ap.add_argument("tickers", nargs="*", help="ticker symbols")
    ap.add_argument("--file", help="file with tickers (whitespace/comma/newline separated)")
    ap.add_argument("--interval", default="1wk", help="yfinance interval (default 1wk)")
    ap.add_argument("--period", default=None, help="yfinance period (default auto by interval)")
    ap.add_argument("--fast", type=int, default=FAST_LEN, help="fast RSI length")
    ap.add_argument("--smooth", type=int, default=SMOOTH, help="RSI smooth length")
    ap.add_argument("--mid", type=float, default=MID_LEVEL, help="zone line (buy only below, sell only above; default 50)")
    ap.add_argument("--ema", type=int, default=200, help="trend EMA length; flags close under it (default 200)")
    ap.add_argument("--csv", help="write results to this CSV file (columns: signal,ticker,fast_rsi,close,ema200,below_ema200)")
    args = ap.parse_args()

    tickers = load_tickers(args)
    if not tickers:
        ap.error("no tickers given (pass symbols or --file)")

    # enough history for slow smoothing at the chosen interval
    period = args.period or {"1wk": "5y", "1d": "2y", "1h": "6mo", "1mo": "max"}.get(args.interval, "5y")

    print(f"Scanning {len(tickers)} tickers | interval={args.interval} period={period} "
          f"fast={args.fast} smooth={args.smooth} mid={args.mid}\n", file=sys.stderr)

    # bulk download (grouped by ticker)
    data = yf.download(tickers, interval=args.interval, period=period,
                       group_by="ticker", auto_adjust=True, progress=False, threads=True)

    buys, sells, errors = [], [], []
    for sym in tickers:
        try:
            if len(tickers) == 1:
                close = data["Close"]
            else:
                close = data[sym]["Close"]
            close = pd.Series(np.asarray(close).ravel(), index=data.index).dropna()
            if len(close) < max(args.fast + args.smooth + 2, 5):
                errors.append((sym, "not enough bars"))
                continue
            sig, rf_now, rf1, rf2 = evaluate(close, args.fast, args.smooth, args.mid)
            ema_val, below = ema200_state(close, args.ema)
            if sig == "BUY":
                buys.append((sym, rf_now, float(close.iloc[-1]), ema_val, below))
            elif sig == "SELL":
                sells.append((sym, rf_now, float(close.iloc[-1]), ema_val, below))
        except Exception as e:  # noqa: BLE001
            errors.append((sym, str(e)[:60]))

    buys.sort(key=lambda x: x[1])       # lowest fast RSI first (deepest valleys)
    sells.sort(key=lambda x: -x[1])     # highest fast RSI first (tallest peaks)

    def flag(below):
        return "↓EMA200" if below else ("      " if below is False else "   ?  ")

    def fmt(rows):
        return "\n".join(
            f"  {s:<8} fastRSI={rf:6.2f}  close={cl:8.2f}  {flag(below)}"
            for s, rf, cl, ev, below in rows
        ) or "  (none)"

    print(f"\n=== BUY signals on current bar ({len(buys)}) ===  (↓EMA200 = close under EMA{args.ema})")
    print(fmt(buys))
    print(f"\n=== SELL signals on current bar ({len(sells)}) ===  (↓EMA200 = close under EMA{args.ema})")
    print(fmt(sells))
    if errors:
        print(f"\n=== skipped/errors ({len(errors)}) ===")
        print("\n".join(f"  {s:<8} {msg}" for s, msg in errors))

    if args.csv:
        rows = ([("BUY", s, rf, cl, ev, below) for s, rf, cl, ev, below in buys]
                + [("SELL", s, rf, cl, ev, below) for s, rf, cl, ev, below in sells])
        df = pd.DataFrame(rows, columns=["signal", "ticker", "fast_rsi", "close", "ema200", "below_ema200"])
        df["fast_rsi"] = df["fast_rsi"].round(2)
        df["close"] = df["close"].round(2)
        df["ema200"] = df["ema200"].round(2)
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {len(df)} signals to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
