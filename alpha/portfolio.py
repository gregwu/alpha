"""Portfolio construction: rank -> select top N with sector caps ->
weight (equal or inverse-vol, position-capped) -> scale gross by regime.

Selection and sizing are deliberately separate steps (the spec's
"separate the decision to buy from the decision to size").
"""

import numpy as np
import pandas as pd

from .config import PORTFOLIO


def select_portfolio(day: pd.DataFrame, regime: str, cfg=PORTFOLIO) -> pd.DataFrame:
    """Build target weights from one day's scored universe.

    `day` needs: ticker, final_score, sector, hv_20. Returns
    ticker/weight rows summing to the regime-scaled gross exposure.
    """
    ranked = day.dropna(subset=["final_score"]).sort_values("final_score", ascending=False)

    max_per_sector = max(1, int(round(cfg.max_sector * cfg.top_n)))
    picks, sector_counts = [], {}
    for row in ranked.itertuples():
        sec = row.sector or "Unknown"
        if sector_counts.get(sec, 0) >= max_per_sector:
            continue
        picks.append(row)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(picks) >= cfg.top_n:
            break
    if not picks:
        return pd.DataFrame(columns=["ticker", "weight"])

    sel = pd.DataFrame({
        "ticker": [p.ticker for p in picks],
        "hv": [getattr(p, "hv_20", np.nan) for p in picks],
    })

    if cfg.weighting == "inverse_vol":
        iv = 1.0 / sel["hv"].clip(lower=0.10)          # floor vol at 10% ann.
        iv = iv.fillna(iv.median())
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(sel), index=sel.index)

    # Position cap with iterative redistribution
    for _ in range(10):
        over = w > cfg.max_position
        if not over.any():
            break
        excess = (w[over] - cfg.max_position).sum()
        w[over] = cfg.max_position
        under = ~over
        if w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break

    gross = cfg.regime_exposure.get(regime, 0.5)
    sel["weight"] = w * gross
    return sel[["ticker", "weight"]]


def rebalance_portfolio(day: pd.DataFrame, regime: str, current: pd.Series,
                        cfg=PORTFOLIO) -> pd.Series:
    """Turnover-aware rebalance: keep held names until their rank decays
    below top_n * keep_buffer, fill freed slots from the top of the list,
    and skip sub-min_trade weight adjustments.

    `current` is the drifted weight series (ticker -> weight). Returns the
    new target weight series.
    """
    ranked = day.dropna(subset=["final_score"]).sort_values("final_score", ascending=False)
    ranked = ranked.reset_index(drop=True)
    rank_of = {t: i + 1 for i, t in enumerate(ranked["ticker"])}
    keep_thresh = int(cfg.top_n * cfg.keep_buffer)
    max_per_sector = max(1, int(round(cfg.max_sector * cfg.top_n)))
    sector_of = dict(zip(ranked["ticker"], ranked["sector"].fillna("Unknown")))
    hv_of = dict(zip(ranked["ticker"], ranked["hv_20"]))

    # 1) keep survivors (still scored and within the rank buffer)
    picks = [t for t in current.index
             if current[t] > 0 and rank_of.get(t, 10**9) <= keep_thresh]
    sector_counts = {}
    for t in picks:
        s = sector_of.get(t, "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    # 2) fill open slots from the top, respecting sector caps
    for row in ranked.itertuples():
        if len(picks) >= cfg.top_n:
            break
        if row.ticker in rank_of and row.ticker not in picks:
            s = sector_of.get(row.ticker, "Unknown")
            if sector_counts.get(s, 0) >= max_per_sector:
                continue
            picks.append(row.ticker)
            sector_counts[s] = sector_counts.get(s, 0) + 1

    if not picks:
        return pd.Series(dtype=float)

    # 3) size positions (same scheme as fresh construction)
    if cfg.weighting == "inverse_vol":
        iv = pd.Series({t: 1.0 / max(hv_of.get(t, np.nan), 0.10) for t in picks})
        iv = iv.fillna(iv.median())
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(picks), index=pd.Index(picks))

    for _ in range(10):
        over = w > cfg.max_position
        if not over.any():
            break
        excess = (w[over] - cfg.max_position).sum()
        w[over] = cfg.max_position
        under = ~over
        if w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break

    gross = cfg.regime_exposure.get(regime, 0.5)
    target = w * gross

    # 4) no-trade band: keep the drifted weight when the adjustment is tiny
    for t in target.index:
        cur = current.get(t, 0.0)
        if cur > 0 and abs(target[t] - cur) < cfg.min_trade:
            target[t] = cur
    return target
