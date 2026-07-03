"""Composite factor score: winsorized cross-sectional z-scores per factor
group, weighted per the research spec.

Fundamentals and options groups are not yet populated with historical
point-in-time data, so their weight is redistributed proportionally
across the available groups (see config.CompositeConfig).
"""

import numpy as np
import pandas as pd

from .config import COMPOSITE
from .features.build import FEATURE_GROUPS

# Direction of each factor within its group: +1 = higher is better.
# Features not listed default to +1. Volatility group: quality-momentum
# systems prefer lower vol / contraction, so most vol features are -1.
FACTOR_SIGNS = {
    "hv_20": -1, "hv_60": -1, "atr14_pct": -1, "parkinson_20": -1,
    "bb_width": -1, "csr_hv_20": -1, "csr_atr14_pct": -1,
    "hv_ratio_20_120": -1, "bb_width_pctile_250": -1, "csr_bb_width_pctile_250": -1,
    "ret_kurt_60": -1, "max_dd_60": 1,      # closer to highs = better
    "bear_bos_20": -1, "bear_fvg_20": -1, "sweep_high_20": -1,
    "dist_swing_high": 1, "days_since_high_60": -1,
    "pct_off_high_60": 1, "pct_off_high_250": 1,
    "beta_60": 0, "spy_corr_60": 0, "ret_skew_60": 0,   # 0 = excluded from composite
    "log_dollar_vol": 0, "csr_log_dollar_vol": 0, "vol_pctile_250": 0,
    "hv_expansion_5": 0, "dist_bear_ob": 0, "di_diff": 1,
    # graph: neighbor momentum propagates to laggards (gap is contrarian)
    "graph_mom_gap_20": -1, "graph_centrality": 0, "graph_avg_corr": 0,
}

# Groups that participate in the composite (momentum folds into trend/RS
# per the spec's weight table).
COMPOSITE_GROUP_MAP = {
    "relative_strength": ["relative_strength"],
    "trend": ["trend", "momentum"],
    "volume": ["volume"],
    "volatility": ["volatility"],
    "structure": ["structure"],
}

# Factor attribution reports on the composite groups plus ML-only families.
ATTRIBUTION_GROUP_MAP = {**COMPOSITE_GROUP_MAP, "graph": ["graph"]}


def _zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Winsorized (±3σ) cross-sectional z-score of each column, by date."""
    z = pd.DataFrame(index=df.index)
    for c in cols:
        by_date = df.groupby("date")[c]
        mu = by_date.transform("mean")
        sd = by_date.transform("std").replace(0, np.nan)
        z[c] = ((df[c] - mu) / sd).clip(-3, 3)
    return z


def composite_score(feat: pd.DataFrame) -> pd.Series:
    """Weighted composite z-score over in-universe rows (NaN elsewhere)."""
    cfg = COMPOSITE
    df = feat[feat["in_universe"]].copy()

    available = {g: w for g, w in cfg.weights.items() if g in COMPOSITE_GROUP_MAP}
    total_w = sum(available.values())

    group_scores = {}
    for group, weight in available.items():
        cols, signs = [], []
        for fam in COMPOSITE_GROUP_MAP[group]:
            for c in FEATURE_GROUPS[fam]:
                s = FACTOR_SIGNS.get(c, 1)
                if s != 0 and c in df.columns:
                    cols.append(c)
                    signs.append(s)
        z = _zscore_by_date(df, cols)
        signed = z * np.array(signs, dtype=float)
        group_scores[group] = signed.mean(axis=1, skipna=True)

    comp = sum((w / total_w) * group_scores[g] for g, w in available.items())
    out = pd.Series(np.nan, index=feat.index, name="composite")
    out.loc[df.index] = comp
    return out


def group_score_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-group z-scores for factor attribution (in-universe rows)."""
    df = feat[feat["in_universe"]].copy()
    out = pd.DataFrame(index=feat.index)
    for group, fams in ATTRIBUTION_GROUP_MAP.items():
        cols, signs = [], []
        for fam in fams:
            for c in FEATURE_GROUPS[fam]:
                s = FACTOR_SIGNS.get(c, 1)
                if s != 0 and c in df.columns:
                    cols.append(c)
                    signs.append(s)
        z = _zscore_by_date(df, cols)
        out.loc[df.index, f"score_{group}"] = (z * np.array(signs, dtype=float)).mean(axis=1)
    return out
