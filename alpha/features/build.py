"""Orchestrate the full feature build: per-ticker families in parallel,
then cross-sectional layers, labels, and the point-in-time universe mask.

Output: data/features.parquet — one row per (ticker, date) with ~130
feature columns, label columns, and `in_universe`.
"""

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..config import DATA_DIR, FEATURES
from ..data import (compute_universe_mask, load_benchmark, load_metadata,
                    load_prices, stock_tickers)
from . import cross_sectional as cs
from . import structure, timeseries

log = logging.getLogger(__name__)

FEATURES_PATH = DATA_DIR / "features.parquet"

# Feature-group membership drives composite scoring and factor attribution.
FEATURE_GROUPS = {
    "trend": [
        "ema9_dist", "ema20_dist", "ema50_dist", "ema200_dist",
        "ema9_slope", "ema20_slope", "ema50_slope", "ema200_slope",
        "sma50_dist", "sma200_dist", "sma50_slope", "sma200_slope",
        "ema20_over_50", "ema50_over_200", "golden_cross", "above_ema200",
        "regslope_20", "regslope_60", "regaccel_20", "regaccel_60",
        "adx14", "di_diff",
    ],
    "momentum": [
        "ret_5", "ret_10", "ret_20", "ret_60", "ret_120", "ret_250", "ret_12_1",
        "mom_consistency_20", "mom_consistency_60", "rsi_5", "rsi_14",
        "pct_off_high_60", "pct_off_low_60", "pct_off_high_250", "days_since_high_60",
        "csr_ret_20", "csr_ret_60", "csr_ret_120", "csr_ret_12_1",
    ],
    "relative_strength": [
        "rs_5", "rs_20", "rs_60", "rs_120", "rs_line_slope_20",
        "beta_60", "spy_corr_60", "csr_rs_20", "csr_rs_60",
        "rs_vs_sector_20", "rs_vs_sector_60",
    ],
    "volume": [
        "log_dollar_vol", "vol_pctile_250", "rel_volume_20", "rel_volume_5v60",
        "obv_slope_20", "vwap20_dist", "cmf_20", "updown_vol_ratio_20",
        "vol_price_corr_20", "csr_log_dollar_vol", "csr_cmf_20", "csr_obv_slope_20",
    ],
    "volatility": [
        "atr14_pct", "hv_20", "hv_60", "hv_ratio_20_120", "hv_expansion_5",
        "parkinson_20", "bb_width", "bb_width_pctile_250", "bb_position",
        "ret_skew_60", "ret_kurt_60", "max_dd_60", "csr_hv_20", "csr_atr14_pct",
        "csr_bb_width_pctile_250",
    ],
    "structure": [
        "dist_swing_high", "dist_swing_low", "bull_bos_20", "bear_bos_20",
        "bos_net_20", "bull_fvg_20", "bear_fvg_20", "last_bull_fvg_size",
        "sweep_low_20", "sweep_high_20", "dist_bull_ob", "dist_bear_ob",
        "range_position_60",
    ],
    "sector": [
        "sector_ret_20", "sector_ret_60", "sector_pct_above_50",
        "sector_pct_above_200", "sector_mom_rank",
    ],
    "graph": [
        "graph_nbr_ret_20", "graph_nbr_rs_20", "graph_nbr_ret_60",
        "graph_mom_gap_20", "graph_centrality", "graph_avg_corr",
    ],
}


def _one_ticker(args):
    ticker, sub = args
    sub = sub.sort_values("date").reset_index(drop=True)
    parts = [
        sub[["ticker", "date", "close", "volume"]],
        timeseries.trend_features(sub),
        timeseries.momentum_features(sub, FEATURES.momentum_windows),
        timeseries.volume_features(sub),
        timeseries.volatility_features(sub),
        structure.structure_features(sub),
    ]
    return pd.concat(parts, axis=1)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Forward returns and their daily cross-sectional percentile rank.

    Labels are NaN near the end of each ticker's history — those rows are
    used for prediction only, never training.
    """
    g = df.groupby("ticker", sort=False)["close"]
    for h in FEATURES.label_horizons:
        fwd = g.shift(-h) / df["close"] - 1.0
        df[f"fwd_ret_{h}"] = fwd
    for h in FEATURES.label_horizons:
        universe_fwd = df[f"fwd_ret_{h}"].where(df["in_universe"])
        df[f"fwd_ret_{h}_rank"] = universe_fwd.groupby(df["date"]).rank(pct=True)
    return df


def build_features(n_workers: int = 6) -> pd.DataFrame:
    prices = load_prices()
    bench = load_benchmark()
    meta = load_metadata()
    log.info("building per-ticker features for %d tickers", prices["ticker"].nunique())

    groups = list(prices.groupby("ticker", sort=False))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_one_ticker, groups, chunksize=25))
    feat = pd.concat(parts, ignore_index=True)
    del parts
    log.info("per-ticker features done: %s", feat.shape)

    feat = feat.sort_values(["ticker", "date"]).reset_index(drop=True)
    feat = cs.relative_strength_features(feat, bench)
    log.info("relative strength done")
    feat = cs.sector_features(feat, meta)
    log.info("sector features done")

    feat["in_universe"] = compute_universe_mask(feat, tradable_tickers=stock_tickers(meta))
    feat = cs.add_cs_ranks(feat)

    from .graph import add_graph_features
    feat = add_graph_features(feat, prices)
    log.info("graph features done")

    feat = add_labels(feat)

    # Compact dtypes to keep the parquet manageable
    float_cols = feat.select_dtypes(include=["float64"]).columns
    feat[float_cols] = feat[float_cols].astype("float32")

    feat.to_parquet(FEATURES_PATH, index=False)
    n_feats = sum(len(v) for v in FEATURE_GROUPS.values())
    log.info("wrote %s: %s (%d grouped features)", FEATURES_PATH, feat.shape, n_feats)
    return feat


def all_feature_columns() -> list[str]:
    return [c for cols in FEATURE_GROUPS.values() for c in cols]
