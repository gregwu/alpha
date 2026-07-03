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
    # Two ranking models blended: the 5d model has the higher IC at both
    # horizons (fast label = more independent training examples), the 20d
    # model diversifies it. Weights are z-score blend weights.
    labels: dict = field(default_factory=lambda: {
        "fwd_ret_20_rank": 0.5,
        "fwd_ret_5_rank": 0.5,
    })
    label: str = "fwd_ret_20_rank"    # primary label (reports/back-compat)
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
    top_n: int = 20
    weighting: str = "inverse_vol"    # "equal" or "inverse_vol"
    max_position: float = 0.05
    max_sector: float = 0.20
    rebalance_freq: str = "W-MON"     # weekly rebalance
    # Turnover controls: hold an existing position until its rank decays
    # below top_n * keep_buffer (asymmetric entry/exit), and skip weight
    # adjustments smaller than min_trade.
    keep_buffer: float = 3.5
    min_trade: float = 0.01
    # Correlation-aware sizing: weight ~ 1/(vol * avg_corr^corr_penalty).
    # Tested 2026-07 and REJECTED (Sharpe 0.54->0.48, MaxDD -39%->-45%
    # as penalty rises 0->1.5): penalizing correlated names tilts into
    # idiosyncratic small-caps whose blowup risk dominates. Keep at 0.
    corr_penalty: float = 0.0
    # Gross exposure by market regime. Goal is to beat SPY, so the default
    # is fully invested: every throttled ladder tested (2026-07) cost more
    # return than it saved (regime gating = risk scaling, not alpha
    # timing). For a drawdown-managed profile use e.g.
    # {"bull":1,"neutral":.7,"high_vol":.5,"bear":.3,"panic":.1}
    # (Sharpe ~same, CAGR -3pts, MaxDD -42%->-27%).
    # bull 1.25 = the lev_trend insight sized to this book: modest leverage
    # only in confirmed uptrends (financing = tbill + spread, charged in
    # the backtest). Tested 2026-07: 1.5x-2x ladders all worse (variance
    # drag + financing on a 26%-vol book); 1.25 was best on excess/IR.
    regime_exposure: dict = field(default_factory=lambda: {
        "bull": 1.25,
        "neutral": 1.00,
        "high_vol": 1.00,
        "bear": 1.00,
        "panic": 1.00,
    })
    cost_bps: float = 10.0            # one-way transaction cost, basis points
    # Financing on gross exposure above 1.0: charged daily at
    # tbill_rate + financing_spread on the borrowed fraction.
    financing_spread: float = 0.01


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
