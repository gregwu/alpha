"""Data layer: snapshot OHLCV from Postgres into parquet, plus benchmark data.

The snapshot decouples research from the live database and makes feature
builds reproducible. Only raw OHLCV is pulled — all features are computed
in this package so every input is point-in-time safe.
"""

import logging

import numpy as np
import pandas as pd
import psycopg2

from .config import DATA_DIR, DB_CONFIG, FEATURES

log = logging.getLogger(__name__)

PRICES_PATH = DATA_DIR / "prices.parquet"
META_PATH = DATA_DIR / "metadata.parquet"
BENCH_PATH = DATA_DIR / "benchmark.parquet"

COARSE_TICKER_SQL = """
    select ticker from stock_data
    where date >= %(start)s
    group by ticker
    having percentile_cont(0.9) within group (order by close_price*volume) > %(min_dv)s
       and max(close_price) > %(min_px)s
"""

OHLCV_SQL = """
    select ticker, date, open_price as open, high_price as high,
           low_price as low, close_price as close, volume
    from stock_data
    where date >= %(start)s and ticker = any(%(tickers)s)
"""


def snapshot_prices(start: str | None = None) -> pd.DataFrame:
    """Pull raw OHLCV for coarsely-liquid tickers into data/prices.parquet."""
    start = start or FEATURES.start_date
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        tickers = pd.read_sql(
            COARSE_TICKER_SQL, conn,
            params={"start": start, "min_dv": 20e6, "min_px": 10.0},
        )["ticker"].tolist()
        log.info("coarse universe: %d tickers", len(tickers))

        df = pd.read_sql(OHLCV_SQL, conn, params={"start": start, "tickers": tickers})
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float64")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df.to_parquet(PRICES_PATH, index=False)
    log.info("wrote %s: %d rows, %d tickers", PRICES_PATH, len(df), df["ticker"].nunique())
    return df


def snapshot_metadata() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        meta = pd.read_sql("select ticker, sector, market_cap from stock_metadata", conn)
    finally:
        conn.close()
    meta["sector"] = meta["sector"].fillna("Unknown").replace("", "Unknown")
    meta.to_parquet(META_PATH, index=False)
    return meta


def snapshot_benchmark(start: str | None = None) -> pd.DataFrame:
    """Fetch SPY and VIX from yfinance -> data/benchmark.parquet."""
    import yfinance as yf

    start = start or FEATURES.start_date
    spy = yf.download("SPY", start=start, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start, progress=False, auto_adjust=True)
    irx = yf.download("^IRX", start=start, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        vix.columns = vix.columns.get_level_values(0)
        irx.columns = irx.columns.get_level_values(0)
    bench = pd.DataFrame({
        "spy_close": spy["Close"],
        "spy_high": spy["High"],
        "spy_low": spy["Low"],
        "spy_volume": spy["Volume"],
        "vix": vix["Close"],
        "tbill_rate": irx["Close"] / 100.0,   # 13wk T-bill, annualized decimal
    })
    bench.index.name = "date"
    bench = bench.reset_index()
    bench["date"] = pd.to_datetime(bench["date"]).dt.tz_localize(None)
    bench[["vix", "tbill_rate"]] = bench[["vix", "tbill_rate"]].ffill()
    bench.to_parquet(BENCH_PATH, index=False)
    log.info("wrote %s: %d rows through %s", BENCH_PATH, len(bench), bench["date"].max().date())
    return bench


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PATH)


def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(META_PATH)


def load_benchmark() -> pd.DataFrame:
    return pd.read_parquet(BENCH_PATH)


def compute_universe_mask(df: pd.DataFrame, cfg=None,
                          tradable_tickers: set | None = None) -> pd.Series:
    """Point-in-time tradability: price and trailing dollar-volume filters.

    Uses only information available on each date (trailing windows).
    `tradable_tickers` restricts to common stocks (ETFs/funds excluded —
    they have no sector classification in stock_metadata).
    """
    from .config import UNIVERSE

    cfg = cfg or UNIVERSE
    g = df.groupby("ticker", sort=False)
    dollar_vol = (df["close"].astype("float64") * df["volume"])
    adv = (
        dollar_vol.groupby(df["ticker"], sort=False)
        .transform(lambda s: s.rolling(cfg.dollar_volume_window, min_periods=20).mean())
    )
    history = g.cumcount()
    mask = (
        (df["close"] >= cfg.min_price)
        & (adv >= cfg.min_dollar_volume)
        & (history >= cfg.min_history_days)
    )
    if tradable_tickers is not None:
        mask &= df["ticker"].isin(tradable_tickers)
    return mask.astype(bool)


def stock_tickers(meta: pd.DataFrame) -> set:
    """Tickers classified as common stocks (have a real sector)."""
    return set(meta.loc[meta["sector"] != "Unknown", "ticker"])
