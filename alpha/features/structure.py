"""ICT / market-structure concepts quantified as numeric features.

Swing points are only "confirmed" K bars after they print, and all levels
are forward-filled from confirmed information, so nothing peeks ahead.
"""

import numpy as np
import pandas as pd

SWING_K = 3  # bars on each side to confirm a swing point


def structure_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype("float64")
    h = df["high"].astype("float64")
    l = df["low"].astype("float64")
    o = df["open"].astype("float64")
    out = pd.DataFrame(index=df.index)
    n = len(df)
    k = SWING_K

    # --- Confirmed swing highs/lows (known only k bars later) ---
    hv, lv = h.values, l.values
    swing_hi = np.full(n, np.nan)
    swing_lo = np.full(n, np.nan)
    if n > 2 * k:
        for i in range(k, n - k):
            win_h = hv[i - k:i + k + 1]
            win_l = lv[i - k:i + k + 1]
            if hv[i] == win_h.max():
                swing_hi[i + k] = hv[i]      # confirmed at i+k
            if lv[i] == win_l.min():
                swing_lo[i + k] = lv[i]
    last_swing_hi = pd.Series(swing_hi, index=df.index).ffill()
    last_swing_lo = pd.Series(swing_lo, index=df.index).ffill()

    out["dist_swing_high"] = c / last_swing_hi - 1.0
    out["dist_swing_low"] = c / last_swing_lo - 1.0

    # --- Break of structure: close crosses the last confirmed swing level ---
    bull_bos = (c > last_swing_hi) & (c.shift() <= last_swing_hi.shift())
    bear_bos = (c < last_swing_lo) & (c.shift() >= last_swing_lo.shift())
    out["bull_bos_20"] = bull_bos.rolling(20).sum()
    out["bear_bos_20"] = bear_bos.rolling(20).sum()
    out["bos_net_20"] = out["bull_bos_20"] - out["bear_bos_20"]

    # --- Fair value gaps (3-candle imbalance) ---
    bull_fvg = l > h.shift(2)
    bear_fvg = h < l.shift(2)
    out["bull_fvg_20"] = bull_fvg.rolling(20).sum()
    out["bear_fvg_20"] = bear_fvg.rolling(20).sum()
    # Size of the most recent bullish FVG relative to price
    gap = (l - h.shift(2)).where(bull_fvg)
    out["last_bull_fvg_size"] = (gap / c).ffill().fillna(0.0).clip(0, 0.2)

    # --- Liquidity sweeps: wick through a prior 20d extreme, close back inside ---
    prior_low = l.rolling(20).min().shift()
    prior_high = h.rolling(20).max().shift()
    sweep_low = (l < prior_low) & (c > prior_low)     # sell-side liquidity taken
    sweep_high = (h > prior_high) & (c < prior_high)  # buy-side liquidity taken
    out["sweep_low_20"] = sweep_low.rolling(20).sum()
    out["sweep_high_20"] = sweep_high.rolling(20).sum()

    # --- Order blocks: opposite-color candle preceding a displacement move ---
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    disp_up = (c - o) > 1.5 * atr
    disp_dn = (o - c) > 1.5 * atr
    bearish_candle = (c < o).shift(fill_value=False)
    bullish_candle = (c > o).shift(fill_value=False)
    bull_ob_level = l.shift().where(disp_up & bearish_candle).ffill()
    bear_ob_level = h.shift().where(disp_dn & bullish_candle).ffill()
    out["dist_bull_ob"] = (c / bull_ob_level - 1.0).clip(-1, 1)
    out["dist_bear_ob"] = (c / bear_ob_level - 1.0).clip(-1, 1)

    # --- Premium/discount within the 60d dealing range ---
    r_hi, r_lo = h.rolling(60).max(), l.rolling(60).min()
    out["range_position_60"] = (c - r_lo) / (r_hi - r_lo).replace(0, np.nan)
    return out
