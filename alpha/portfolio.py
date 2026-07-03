"""Portfolio construction: rank -> select top N with sector caps ->
weight (equal or inverse-vol, position-capped) -> scale gross by regime.

Selection and sizing are deliberately separate steps (the spec's
"separate the decision to buy from the decision to size").
"""

import numpy as np
import pandas as pd

from .config import PORTFOLIO


def _risk_weights(hv: pd.Series, corr: pd.Series, cfg=PORTFOLIO) -> pd.Series:
    """Inverse marginal-risk weights: 1 / (vol * avg_corr^penalty), capped.

    Marginal contribution to portfolio risk is ~ sigma_i * rho_i, so this
    approximates equal risk contribution and de-concentrates correlated
    clusters. Falls back to plain inverse-vol when corr is missing.
    """
    if cfg.weighting == "equal":
        w = pd.Series(1.0 / len(hv), index=hv.index)
    else:
        sigma = hv.clip(lower=0.10)
        sigma = sigma.fillna(sigma.median())
        rho = corr.clip(lower=0.10, upper=1.0) ** cfg.corr_penalty
        rho = rho.fillna(rho.median() if rho.notna().any() else 1.0)
        iv = 1.0 / (sigma * rho)
        w = iv / iv.sum()

    # Waterfall cap: fix names at the cap, rescale the rest to the leftover
    # budget; converges even when the cap binds for everyone.
    capped = pd.Series(False, index=w.index)
    for _ in range(len(w)):
        over = (w > cfg.max_position) & ~capped
        if not over.any():
            break
        capped |= over
        w[capped] = cfg.max_position
        free = ~capped
        budget = 1.0 - capped.sum() * cfg.max_position
        if budget <= 0 or w[free].sum() <= 0:
            break
        w[free] *= budget / w[free].sum()
    return w


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
        "corr": [getattr(p, "graph_avg_corr", np.nan) for p in picks],
    })
    w = _risk_weights(sel["hv"], sel["corr"], cfg)

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
    corr_col = ranked["graph_avg_corr"] if "graph_avg_corr" in ranked.columns \
        else pd.Series(np.nan, index=ranked.index)
    corr_of = dict(zip(ranked["ticker"], corr_col))

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
    idx = pd.Index(picks)
    hv = pd.Series({t: hv_of.get(t, np.nan) for t in picks}, index=idx)
    corr = pd.Series({t: corr_of.get(t, np.nan) for t in picks}, index=idx)
    w = _risk_weights(hv, corr, cfg)

    gross = cfg.regime_exposure.get(regime, 0.5)
    target = w * gross

    # 4) no-trade band: keep the drifted weight when the adjustment is tiny
    for t in target.index:
        cur = current.get(t, 0.0)
        if cur > 0 and abs(target[t] - cur) < cfg.min_trade:
            target[t] = cur
    return target
