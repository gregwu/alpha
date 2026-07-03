"""Replicate the tradingview-mcp bots on alpha's data and evaluate them
honestly.

Bots (from backtest_beat_spy.py, weekly bars, next-bar fills):
  lev_trend : 2x SPY (synthetic, -1%/yr drag) while SPY > 40wk SMA, else cash
  dual_mom  : 4-weekly rotation SPY/EFA/TLT by 52wk momentum, floor TLT
  momentum  : top-15 by 26wk momentum, equal weight, 4-weekly rebalance

For the momentum bot we run three variants to decompose the edge:
  current-SP500, no costs   — faithful replication (survivorship-biased)
  current-SP500, 10bps      — costs only
  point-in-time universe, 10bps — the honest number (delisted names included)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

SMA_WK, MOM_WK, DUAL_WK, LEV, LEV_COST = 40, 26, 52, 2.0, 0.01
COST = 10 / 1e4   # one-way

WINDOWS = [("5y", "2021-07-02", None), ("2022", "2022-01-01", "2022-12-31"),
           ("8y", "2018-07-02", None), ("2016+", "2016-01-04", None)]


def stats(weekly_ret):
    r = weekly_ret.dropna()
    if len(r) < 10:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    return {
        "total_pct": (eq.iloc[-1] - 1) * 100,
        "cagr_pct": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
        "max_dd_pct": (eq / eq.cummax() - 1).min() * 100,
        "sharpe": r.mean() / (r.std() + 1e-12) * np.sqrt(52),
    }


def lev_trend_returns(spy_w):
    sma = spy_w.rolling(SMA_WK).mean()
    in_mkt = (spy_w > sma).shift(1).fillna(False)
    r = spy_w.pct_change(fill_method=None)
    ret = pd.Series(np.where(in_mkt, r * LEV - LEV_COST / 52, 0.0), index=spy_w.index)
    switches = in_mkt.astype(int).diff().abs().fillna(0)
    return ret, switches * COST  # cost when entering/exiting the ETF


def dual_mom_returns(px_w):
    mom = px_w / px_w.shift(DUAL_WK) - 1.0
    rets = px_w.pct_change(fill_method=None)
    holding, out, costs = "TLT", [], []
    prev = holding
    for i in range(len(px_w)):
        if i > 0 and i % 4 == 0 and i >= DUAL_WK:
            m = mom.iloc[i - 1].dropna()
            if len(m):
                best = m.idxmax()
                holding = best if m[best] > 0 else "TLT"
        r = rets.iloc[i].get(holding, np.nan) if i > 0 else 0.0
        out.append(0.0 if pd.isna(r) else r)
        costs.append(COST * 2 if holding != prev else 0.0)  # sell one, buy other
        prev = holding
    return (pd.Series(out, index=px_w.index),
            pd.Series(costs, index=px_w.index))


def momentum_returns(px_w, eligible_w=None):
    """Top-15 by 26wk momentum, equal weight, 4-weekly. `eligible_w` is an
    optional (week x ticker) bool frame restricting each decision to the
    point-in-time universe."""
    rets = px_w.pct_change(fill_method=None)
    mom = px_w / px_w.shift(MOM_WK) - 1.0
    holdings, out, costs = [], [], []
    for i in range(len(px_w)):
        r = rets.iloc[i][holdings].mean(skipna=True) if (i > 0 and holdings) else 0.0
        out.append(0.0 if pd.isna(r) else r)
        cost = 0.0
        if i >= MOM_WK and i % 4 == 0:
            m = mom.iloc[i - 1].dropna()
            if eligible_w is not None:
                elig = eligible_w.iloc[max(i - 1, 0)]
                m = m[m.index.isin(elig.index[elig])]
            if len(m):
                new = list(m.nlargest(15).index)
                changed = len(set(new) - set(holdings))
                cost = (changed / 15) * COST * 2 if holdings else COST
                holdings = new
        out[-1] = out[-1]
        costs.append(cost)
    return (pd.Series(out, index=px_w.index),
            pd.Series(costs, index=px_w.index))


def main():
    prices = pd.read_parquet("data/prices.parquet", columns=["ticker", "date", "close"])
    px = prices.pivot_table(index="date", columns="ticker", values="close")
    px_w = px.resample("W-FRI").last()

    spy_w = px_w["SPY.US"].dropna()
    spy_ret_w = spy_w.pct_change(fill_method=None)

    sp500 = [l.strip() + ".US" for l in
             open("/Users/gangwu/git/tradingview-mcp/sp500.txt") if l.strip()]
    sp500 = [t for t in sp500 if t in px_w.columns]

    uni = pd.read_parquet("data/features.parquet",
                          columns=["ticker", "date", "in_universe"])
    uni_w = (uni.pivot_table(index="date", columns="ticker", values="in_universe",
                             aggfunc="last").fillna(False).astype(bool)
             .resample("W-FRI").last().reindex(px_w.index).fillna(False))

    lev_r, lev_c = lev_trend_returns(spy_w)
    dm_r, dm_c = dual_mom_returns(px_w[["SPY.US", "EFA.US", "TLT.US"]].dropna())
    mo_sp_r, mo_sp_c = momentum_returns(px_w[sp500])
    pit_px = px_w[uni_w.columns.intersection(px_w.columns)]
    mo_pit_r, mo_pit_c = momentum_returns(pit_px, eligible_w=uni_w)

    strategies = {
        "lev_trend (replication)": lev_r,
        "lev_trend + costs": lev_r - lev_c,
        "dual_mom (replication)": dm_r,
        "dual_mom + costs": dm_r - dm_c,
        "momentum current-SP500 (replication)": mo_sp_r,
        "momentum current-SP500 + costs": mo_sp_r - mo_sp_c,
        "momentum PIT universe + costs": mo_pit_r - mo_pit_c,
        "SPY buy&hold": spy_ret_w,
    }

    rows = []
    for wname, start, end in WINDOWS:
        for name, r in strategies.items():
            rr = r.loc[start:end] if end else r.loc[start:]
            s = stats(rr)
            if s:
                rows.append({"window": wname, "strategy": name,
                             **{k: round(v, 2) for k, v in s.items()}})
    df = pd.DataFrame(rows)
    df.to_csv("reports/bot_replication.csv", index=False)
    for wname in df["window"].unique():
        print(f"\n=== {wname} ===")
        print(df[df.window == wname].drop(columns="window")
                .sort_values("cagr_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
