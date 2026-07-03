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
