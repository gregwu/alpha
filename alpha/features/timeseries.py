"""Per-ticker time-series features: trend, momentum, volume, volatility.

Each function takes a single ticker's OHLCV DataFrame (sorted by date) and
returns a DataFrame of feature columns aligned to the same index. All
windows are trailing — no look-ahead.
"""

import numpy as np
import pandas as pd


def _slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope, normalized by the level (pct/day)."""
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    denom = (x ** 2).sum()

    def f(y):
        return float(np.dot(x, y)) / denom

    raw = s.rolling(window).apply(f, raw=True, engine="numba") if len(s) > 5000 else \
        s.rolling(window).apply(f, raw=True)
    return raw / s


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype("float64")
    out = pd.DataFrame(index=df.index)

    emas = {n: _ema(c, n) for n in (9, 20, 50, 200)}
    smas = {n: c.rolling(n).mean() for n in (50, 200)}

    for n, e in emas.items():
        out[f"ema{n}_dist"] = c / e - 1.0
        out[f"ema{n}_slope"] = e.pct_change(5, fill_method=None) / 5.0
    for n, m in smas.items():
        out[f"sma{n}_dist"] = c / m - 1.0
        out[f"sma{n}_slope"] = m.pct_change(5, fill_method=None) / 5.0

    out["ema20_over_50"] = emas[20] / emas[50] - 1.0
    out["ema50_over_200"] = emas[50] / emas[200] - 1.0
    out["golden_cross"] = (smas[50] > smas[200]).astype("float32")
    out["above_ema200"] = (c > emas[200]).astype("float32")

    # Rolling regression slope + acceleration (change in slope)
    for w in (20, 60):
        sl = _slope(c, w)
        out[f"regslope_{w}"] = sl
        out[f"regaccel_{w}"] = sl.diff(5)

    # ADX(14)
    h, l = df["high"].astype("float64"), df["low"].astype("float64")
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["di_diff"] = pdi - mdi
    return out


def momentum_features(df: pd.DataFrame, windows=(5, 10, 20, 60, 120, 250)) -> pd.DataFrame:
    c = df["close"].astype("float64")
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"ret_{w}"] = c.pct_change(w, fill_method=None)
    out["ret_12_1"] = c.shift(20) / c.shift(250) - 1.0   # classic 12-1 momentum
    out["mom_consistency_20"] = (c.pct_change(fill_method=None) > 0).rolling(20).mean()
    out["mom_consistency_60"] = (c.pct_change(fill_method=None) > 0).rolling(60).mean()

    # RSI at multiple speeds
    delta = c.diff()
    for w in (5, 14):
        gain = delta.clip(lower=0).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    out["pct_off_high_60"] = c / c.rolling(60).max() - 1.0
    out["pct_off_low_60"] = c / c.rolling(60).min() - 1.0
    out["pct_off_high_250"] = c / c.rolling(250).max() - 1.0
    out["days_since_high_60"] = (
        c.rolling(60).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True)
    )
    return out


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype("float64")
    v = df["volume"].astype("float64")
    h, l = df["high"].astype("float64"), df["low"].astype("float64")
    out = pd.DataFrame(index=df.index)

    dv = c * v
    out["log_dollar_vol"] = np.log1p(dv.rolling(20).mean())
    out["vol_pctile_250"] = v.rolling(250).rank(pct=True)
    out["rel_volume_20"] = v / v.rolling(20).mean()
    out["rel_volume_5v60"] = v.rolling(5).mean() / v.rolling(60).mean()

    # OBV slope (normalized by average volume so it's comparable cross-stock)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_slope_20"] = obv.diff(20) / (20 * v.rolling(60).mean())

    # Rolling VWAP distance (20-day)
    tp = (h + l + c) / 3
    vwap20 = (tp * v).rolling(20).sum() / v.rolling(20).sum()
    out["vwap20_dist"] = c / vwap20 - 1.0

    # Accumulation: Chaikin Money Flow + up/down volume ratio
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    out["cmf_20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum()
    up_vol = v.where(c.diff() > 0, 0.0).rolling(20).sum()
    dn_vol = v.where(c.diff() < 0, 0.0).rolling(20).sum()
    out["updown_vol_ratio_20"] = up_vol / dn_vol.replace(0, np.nan)

    # Volume-confirmed moves: correlation of |return| with relative volume
    out["vol_price_corr_20"] = c.pct_change(fill_method=None).rolling(20).corr(v.pct_change(fill_method=None))
    return out


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype("float64")
    h, l = df["high"].astype("float64"), df["low"].astype("float64")
    r = c.pct_change(fill_method=None)
    out = pd.DataFrame(index=df.index)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr14_pct"] = atr14 / c

    for w in (20, 60):
        out[f"hv_{w}"] = r.rolling(w).std() * np.sqrt(252)
    out["hv_ratio_20_120"] = r.rolling(20).std() / r.rolling(120).std()  # contraction<1
    out["hv_expansion_5"] = r.rolling(20).std().pct_change(5, fill_method=None)

    # Parkinson (high-low) volatility
    hl = np.log(h / l.replace(0, np.nan)) ** 2
    out["parkinson_20"] = np.sqrt(hl.rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

    # Bollinger band width + position
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_width"] = 4 * s20 / m20
    out["bb_width_pctile_250"] = out["bb_width"].rolling(250).rank(pct=True)
    out["bb_position"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)

    out["ret_skew_60"] = r.rolling(60).skew()
    out["ret_kurt_60"] = r.rolling(60).kurt()
    out["max_dd_60"] = c / c.rolling(60).max() - 1.0
    return out
