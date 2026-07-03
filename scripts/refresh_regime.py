"""Recompute the market-regime series from the current feature matrix."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd

from alpha.data import load_benchmark
from alpha.features.build import FEATURES_PATH
from alpha.regime import classify_regime, compute_breadth

if __name__ == "__main__":
    feat = pd.read_parquet(
        FEATURES_PATH,
        columns=["ticker", "date", "in_universe", "above_ema200", "sma50_dist",
                 "pct_off_high_60", "pct_off_low_60"],
    )
    reg = classify_regime(load_benchmark(), compute_breadth(feat))
    print(f"regime through {reg['date'].max().date()}: {reg['regime'].iloc[-1]}")
