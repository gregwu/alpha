"""Cross-sectional features: relative strength vs SPY, beta, sector
aggregates, and daily cross-sectional ranks of key signals.
"""

import numpy as np
import pandas as pd

RS_WINDOWS = (5, 20, 60, 120)


def relative_strength_features(feat: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Add excess-return-vs-SPY, RS-line slope, rolling beta/correlation.

    `feat` is the tall frame with ticker/date and ret_* columns already built.
    """
    b = bench.set_index("date")["spy_close"].astype("float64")
    spy_ret = {w: (b / b.shift(w) - 1.0).rename(f"spy_ret_{w}") for w in RS_WINDOWS}
    spy_daily = b.pct_change(fill_method=None).rename("spy_ret_1d")

    merged = feat.merge(
        pd.concat(list(spy_ret.values()) + [spy_daily], axis=1).reset_index(),
        on="date", how="left",
    )
    for w in RS_WINDOWS:
        merged[f"rs_{w}"] = merged[f"ret_{w}"] - merged[f"spy_ret_{w}"]
        merged.drop(columns=[f"spy_ret_{w}"], inplace=True)

    # RS line slope: 20d change of cumulative excess return, per ticker
    daily_ret = merged.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    excess = daily_ret - merged["spy_ret_1d"]
    g = excess.groupby(merged["ticker"], sort=False)
    merged["rs_line_slope_20"] = g.transform(lambda s: s.rolling(20).mean())

    # Rolling 60d beta and correlation to SPY
    merged["_r"] = daily_ret
    grp = merged.groupby("ticker", sort=False)

    def per_ticker(dfg):
        cov = dfg["_r"].rolling(60).cov(dfg["spy_ret_1d"])
        var = dfg["spy_ret_1d"].rolling(60).var()
        beta = cov / var
        corr = dfg["_r"].rolling(60).corr(dfg["spy_ret_1d"])
        return pd.DataFrame({"beta_60": beta, "spy_corr_60": corr}, index=dfg.index)

    stats = grp[["_r", "spy_ret_1d"]].apply(per_ticker)
    stats = stats.reset_index(level=0, drop=True)
    merged[["beta_60", "spy_corr_60"]] = stats
    merged.drop(columns=["_r", "spy_ret_1d"], inplace=True)
    return merged


def sector_features(feat: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Sector relative strength, breadth (pct above 50/200 MA), momentum rank."""
    df = feat.merge(meta[["ticker", "sector"]], on="ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")

    gb = df.groupby(["date", "sector"], sort=False)
    sec = gb.agg(
        sector_ret_20=("ret_20", "mean"),
        sector_ret_60=("ret_60", "mean"),
        sector_pct_above_50=("sma50_dist", lambda s: (s > 0).mean()),
        sector_pct_above_200=("sma200_dist", lambda s: (s > 0).mean()),
    ).reset_index()

    # Sector momentum rank across sectors each day
    sec["sector_mom_rank"] = sec.groupby("date")["sector_ret_60"].rank(pct=True)

    df = df.merge(sec, on=["date", "sector"], how="left")
    df["rs_vs_sector_20"] = df["ret_20"] - df["sector_ret_20"]
    df["rs_vs_sector_60"] = df["ret_60"] - df["sector_ret_60"]
    return df


RANK_COLS = [
    "ret_20", "ret_60", "ret_120", "ret_12_1", "rs_20", "rs_60",
    "hv_20", "atr14_pct", "log_dollar_vol", "cmf_20", "obv_slope_20",
    "regslope_20", "bb_width_pctile_250",
]


def add_cs_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional percentile ranks of key raw signals."""
    for col in RANK_COLS:
        if col in df.columns:
            df[f"csr_{col}"] = df.groupby("date")[col].rank(pct=True)
    return df
