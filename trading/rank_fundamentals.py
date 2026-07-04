#!/usr/bin/env python3
"""
rank_fundamentals.py — rank a ticker list by SEC-EDGAR fundamentals.

Replaces the earlier one-off ranker that used cached TradingView-scanner data.
Fundamentals now come from edgar_fundamentals.py (authoritative SEC XBRL).

Score blends (z-scored, higher = better):
    growth   : eps_yoy + revenue_yoy
    quality  : roe + gross_margin
    value    : lower P/E is better
    leverage : lower debt/equity is better

Usage:
    python3 rank_fundamentals.py --file sp500.txt --top 20
    python3 rank_fundamentals.py AAPL MSFT NVDA EQT --csv report/fund_rank.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import edgar_fundamentals as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--csv")
    args = ap.parse_args()

    syms = list(args.tickers)
    if args.file:
        with open(args.file) as f:
            for line in f:
                for tok in line.replace(",", " ").split():
                    if tok.strip() and not tok.startswith("#"):
                        syms.append(tok.strip().upper())
    syms = list(dict.fromkeys(s.upper() for s in syms))
    if not syms:
        ap.error("no tickers")

    # prices for P/E
    print(f"Fetching prices for {len(syms)} tickers...", file=sys.stderr)
    px = yf.download(syms, period="5d", interval="1d", progress=False)["Close"]
    prices = {}
    if len(syms) == 1:
        prices[syms[0]] = float(px.dropna().iloc[-1])
    else:
        for s in px.columns:
            col = px[s].dropna()
            if len(col):
                prices[s] = float(col.iloc[-1])

    cik = E.ticker_to_cik()
    rows = [E.get_fundamentals(s, cik, price=prices.get(s)) for s in syms]
    df = pd.DataFrame(rows)
    df = df[df.get("eps_ttm").notna()] if "eps_ttm" in df else df

    def z(col, higher=True):
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.clip(s.quantile(0.02), s.quantile(0.98))
        zz = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zz if higher else -zz

    growth = z("eps_yoy") + z("revenue_yoy")
    quality = z("roe") + z("gross_margin")
    value = z("pe", higher=False)
    lever = z("debt_to_equity", higher=False)
    df["score"] = (growth.fillna(0) + quality.fillna(0) + 0.8 * value.fillna(0)
                   + 0.5 * lever.fillna(0)).round(2)
    df = df.sort_values("score", ascending=False)

    show = df[["ticker", "score", "pe", "eps_ttm", "eps_yoy", "revenue_yoy",
               "roe", "gross_margin", "debt_to_equity"]].head(args.top)
    pd.set_option("display.width", 200)
    print(f"\n=== Top {args.top} by SEC-EDGAR fundamentals ===")
    print(show.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {len(df)} ranked rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
