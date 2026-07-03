"""Central configuration for the alpha platform."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)

# Reuses the same Postgres instance as pattern_scan / yahoostock.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5433)),
    "dbname": os.getenv("DB_NAME", "database"),
    "user": os.getenv("DB_USER", "user"),
    "password": os.getenv("DB_PASSWORD", "password"),
}


@dataclass
class UniverseConfig:
    min_price: float = 10.0
    min_dollar_volume: float = 20e6   # 63-day average daily dollar volume
    dollar_volume_window: int = 63
    min_history_days: int = 300       # need enough bars for 250d features
    # Market-cap filter is applied only in live ranking (metadata is
    # current-snapshot, not point-in-time; using it historically would
    # introduce survivorship bias).
    min_market_cap: float = 1e9


@dataclass
class FeatureConfig:
    start_date: str = "2012-01-01"    # feature history start (warmup included)
    momentum_windows: tuple = (5, 10, 20, 60, 120, 250)
    rs_windows: tuple = (5, 20, 60, 120)
    label_horizons: tuple = (5, 20)   # forward-return horizons in trading days


@dataclass
class ModelConfig:
    label: str = "fwd_ret_20_rank"    # cross-sectional rank of 20d forward return
    train_years: int = 4              # trailing training window
    retrain_freq: str = "MS"          # month start
    embargo_days: int = 21            # purge gap between train end and prediction
    max_train_rows: int = 1_200_000   # subsample cap per fold
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "num_threads": 0,
    })
    num_boost_round: int = 300


@dataclass
class PortfolioConfig:
    top_n: int = 30
    weighting: str = "inverse_vol"    # "equal" or "inverse_vol"
    max_position: float = 0.05
    max_sector: float = 0.20
    rebalance_freq: str = "W-MON"     # weekly rebalance
    # Gross exposure by market regime (dynamic cash allocation).
    regime_exposure: dict = field(default_factory=lambda: {
        "bull": 1.00,
        "neutral": 0.70,
        "high_vol": 0.50,
        "bear": 0.30,
        "panic": 0.10,
    })
    cost_bps: float = 10.0            # one-way transaction cost, basis points


@dataclass
class BacktestConfig:
    start: str = "2016-01-01"
    end: str | None = None            # None = last available date


@dataclass
class CompositeConfig:
    # Factor-group weights per the research spec; ML gets blended on top.
    weights: dict = field(default_factory=lambda: {
        "relative_strength": 0.25,
        "trend": 0.20,
        "volume": 0.15,
        "fundamentals": 0.15,   # redistributed if fundamentals unavailable
        "volatility": 0.10,
        "structure": 0.10,
        "options": 0.05,        # redistributed if options unavailable
    })
    ml_blend: float = 0.5       # final = ml_blend*ML_z + (1-ml_blend)*composite_z


UNIVERSE = UniverseConfig()
FEATURES = FeatureConfig()
MODEL = ModelConfig()
PORTFOLIO = PortfolioConfig()
BACKTEST = BacktestConfig()
COMPOSITE = CompositeConfig()
