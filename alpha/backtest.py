"""Walk-forward backtest of the full pipeline.

Timing discipline: scores use features as of close t-1; the portfolio is
executed at the close of rebalance day t. Transaction costs are charged
on turnover at each rebalance.
"""

import logging

import numpy as np
import pandas as pd

from .config import BACKTEST, COMPOSITE, DATA_DIR, PORTFOLIO
from .portfolio import rebalance_portfolio

log = logging.getLogger(__name__)

RESULT_PATH = DATA_DIR / "backtest_daily.parquet"
HOLDINGS_PATH = DATA_DIR / "backtest_holdings.parquet"


def _zscore_by_date(s: pd.Series, dates: pd.Series) -> pd.Series:
    mu = s.groupby(dates).transform("mean")
    sd = s.groupby(dates).transform("std")
    return (s - mu) / sd.replace(0, np.nan)


def build_scores(feat: pd.DataFrame, preds: pd.DataFrame,
                 composite: pd.Series) -> pd.DataFrame:
    """Blend ML score and composite into final_score per (ticker, date)."""
    cols = ["ticker", "date", "sector", "hv_20", "close"]
    if "graph_avg_corr" in feat.columns:
        cols.append("graph_avg_corr")
    df = feat.loc[feat["in_universe"], cols].copy()
    df["composite"] = composite.loc[df.index]
    df = df.merge(preds, on=["ticker", "date"], how="left")

    df["ml_z"] = _zscore_by_date(df["ml_score"], df["date"])
    df["comp_z"] = _zscore_by_date(df["composite"], df["date"])
    b = COMPOSITE.ml_blend
    df["final_score"] = np.where(
        df["ml_z"].notna(), b * df["ml_z"] + (1 - b) * df["comp_z"], df["comp_z"],
    )
    return df


def run_backtest(scores: pd.DataFrame, prices: pd.DataFrame, regime: pd.DataFrame,
                 bench: pd.DataFrame, start=None, end=None,
                 pcfg=PORTFOLIO) -> pd.DataFrame:
    start = pd.Timestamp(start or BACKTEST.start)
    end = pd.Timestamp(end) if end or BACKTEST.end else scores["date"].max()

    # Daily returns matrix (date x ticker)
    px = prices.pivot_table(index="date", columns="ticker", values="close")
    rets = px.pct_change(fill_method=None)

    trading_days = px.index[(px.index >= start) & (px.index <= end)]
    # Rebalance on the first trading day of each week
    week = pd.Series(trading_days).dt.isocalendar()
    week_key = week["year"].astype(str) + "-" + week["week"].astype(str)
    rebal_days = pd.DatetimeIndex(pd.Series(trading_days).groupby(week_key.values).min())
    rebal_days = rebal_days.sort_values()

    regime_by_date = regime.set_index("date")["regime"]
    # Daily financing rate for leverage (T-bill + spread), forward-filled
    rate = (bench.set_index("date")["tbill_rate"]
            if "tbill_rate" in bench.columns else pd.Series(dtype=float))
    daily_rate = (rate.reindex(px.index).ffill().fillna(0.03)
                  + pcfg.financing_spread) / 252.0
    scores_by_date = dict(tuple(scores.groupby("date", sort=True)))
    score_dates = np.array(sorted(scores_by_date.keys()))

    weights = pd.Series(dtype=float)      # current holdings (ticker -> weight)
    rows, holdings_rows = [], []
    cost_rate = pcfg.cost_bps / 1e4
    rebal_set = set(rebal_days)

    for day in trading_days:
        # 1) apply today's market moves to yesterday's holdings
        gross_ret = 0.0
        if len(weights):
            r = rets.loc[day].reindex(weights.index).fillna(0.0)
            gross_ret = float((weights * r).sum())
            # drift position weights as fractions of total portfolio value
            weights = weights * (1 + r) / (1 + gross_ret)

        cost = 0.0
        if day in rebal_set:
            # 2) rebalance at today's close using scores from the prior close
            prior = score_dates[score_dates < np.datetime64(day)]
            if len(prior):
                sday = scores_by_date[pd.Timestamp(prior[-1])]
                reg = regime_by_date.asof(day) if day not in regime_by_date.index \
                    else regime_by_date.loc[day]
                if isinstance(reg, pd.Series):
                    reg = reg.iloc[-1]
                target = rebalance_portfolio(sday, reg or "neutral", weights, pcfg)
                turnover = (target.reindex(weights.index.union(target.index), fill_value=0)
                            - weights.reindex(weights.index.union(target.index), fill_value=0)
                            ).abs().sum()
                cost = turnover * cost_rate
                weights = target
                holdings_rows.append(
                    pd.DataFrame({"date": day, "ticker": target.index,
                                  "weight": target.values, "regime": reg}))

        financing = max(float(weights.sum()) - 1.0, 0.0) * float(daily_rate.get(day, 0.0))
        net_ret = gross_ret - cost - financing
        rows.append({"date": day, "ret": net_ret, "gross_exposure": float(weights.sum()),
                     "n_holdings": int((weights > 0).sum()), "cost": cost,
                     "financing": financing})

    daily = pd.DataFrame(rows).set_index("date")
    spy = bench.set_index("date")["spy_close"].pct_change(fill_method=None)
    daily["spy_ret"] = spy.reindex(daily.index).fillna(0.0)
    daily["equity"] = (1 + daily["ret"]).cumprod()
    daily["spy_equity"] = (1 + daily["spy_ret"]).cumprod()

    daily.reset_index().to_parquet(RESULT_PATH, index=False)
    if holdings_rows:
        pd.concat(holdings_rows, ignore_index=True).to_parquet(HOLDINGS_PATH, index=False)
    log.info("backtest %s..%s: final equity %.2fx (SPY %.2fx)",
             start.date(), end.date(), daily["equity"].iloc[-1], daily["spy_equity"].iloc[-1])
    return daily
