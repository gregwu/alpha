"""Market regime classification: bull / neutral / bear / high_vol / panic.

Inputs: SPY trend and drawdown, VIX level, realized volatility, and
market breadth computed from the stock universe. Rule-based and fully
transparent — the regime gates gross exposure, it does not pick stocks.
"""

import numpy as np
import pandas as pd

from .config import DATA_DIR

REGIME_PATH = DATA_DIR / "regime.parquet"

REGIMES = ["bull", "neutral", "bear", "high_vol", "panic"]


def compute_breadth(feat: pd.DataFrame) -> pd.DataFrame:
    """Universe breadth by date: % above 200d EMA, % above 50d SMA,
    new 60d highs minus lows (as % of names)."""
    u = feat[feat["in_universe"]]
    g = u.groupby("date")
    breadth = pd.DataFrame({
        "pct_above_200": g["above_ema200"].mean(),
        "pct_above_50": g["sma50_dist"].apply(lambda s: (s > 0).mean()),
        "new_high_60": g["pct_off_high_60"].apply(lambda s: (s > -0.001).mean()),
        "new_low_60": g["pct_off_low_60"].apply(lambda s: (s < 0.001).mean()),
    })
    breadth["hl_net_60"] = breadth["new_high_60"] - breadth["new_low_60"]
    return breadth.reset_index()


def classify_regime(bench: pd.DataFrame, breadth: pd.DataFrame) -> pd.DataFrame:
    """Daily regime label from SPY/VIX/breadth. All inputs are trailing."""
    df = bench.copy().sort_values("date")
    spy = df["spy_close"].astype("float64")
    df["spy_sma200"] = spy.rolling(200).mean()
    df["spy_ret_20"] = spy.pct_change(20, fill_method=None)
    df["spy_dd"] = spy / spy.rolling(250).max() - 1.0
    df["rv_20"] = spy.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)

    df = df.merge(breadth, on="date", how="left")
    df[["pct_above_200", "hl_net_60"]] = df[["pct_above_200", "hl_net_60"]].ffill()

    above_200 = df["spy_close"] > df["spy_sma200"]
    vix = df["vix"]

    conditions = [
        # panic: crash conditions — very high vol or deep fast drawdown
        (vix >= 35) | (df["spy_dd"] <= -0.18) | (df["rv_20"] >= 0.40),
        # bear: below 200dma with weak breadth
        (~above_200) & ((df["pct_above_200"] < 0.40) | (df["spy_dd"] <= -0.10)),
        # high_vol: elevated vol but trend intact
        (vix >= 25) | (df["rv_20"] >= 0.25),
        # bull: above 200dma with healthy breadth and momentum
        above_200 & (df["pct_above_200"] >= 0.55) & (df["spy_ret_20"] > -0.02),
    ]
    choices = ["panic", "bear", "high_vol", "bull"]
    df["regime"] = np.select(conditions, choices, default="neutral")

    # Debounce: only switch after the raw label persists 3 consecutive days
    raw = df["regime"].tolist()
    smoothed = raw[:1]
    for i in range(1, len(raw)):
        if raw[i] == "panic":  # derisk immediately, no debounce
            smoothed.append("panic")
        elif raw[i] != smoothed[-1] and i >= 2 and raw[i] == raw[i - 1] == raw[i - 2]:
            smoothed.append(raw[i])
        else:
            smoothed.append(smoothed[-1])
    df["regime_raw"] = df["regime"]
    df["regime"] = smoothed

    out = df[["date", "regime", "regime_raw", "spy_close", "spy_dd", "rv_20",
              "vix", "pct_above_200", "hl_net_60"]]
    out.to_parquet(REGIME_PATH, index=False)
    return out


def load_regime() -> pd.DataFrame:
    return pd.read_parquet(REGIME_PATH)
