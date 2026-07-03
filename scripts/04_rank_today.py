"""Produce today's ranked stock list and target portfolio from the most
recent data in the feature matrix, using the latest trained model.

Run scripts/01..03 first (and re-sync the stock_data table so prices are
current). Output: reports/rank_YYYY-MM-DD.csv + printed target portfolio.

Note: the live universe additionally applies the market-cap filter
(>$1B) from stock_metadata, which is skipped historically to avoid
survivorship bias.
"""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import numpy as np
import pandas as pd

from alpha.backtest import _zscore_by_date
from alpha.composite import composite_score
from alpha.config import COMPOSITE, MODELS_DIR, REPORTS_DIR, UNIVERSE
from alpha.data import load_benchmark, load_metadata
from alpha.features.build import FEATURES_PATH
from alpha.portfolio import select_portfolio
from alpha.regime import load_regime

if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    meta = load_metadata()
    asof = feat["date"].max()
    day = feat[(feat["date"] == asof) & feat["in_universe"]].copy()
    print(f"as-of date: {asof.date()}  universe: {len(day)}")

    # Live-only market-cap filter
    mcap = meta.set_index("ticker")["market_cap"]
    day["market_cap"] = day["ticker"].map(mcap)
    day = day[day["market_cap"].fillna(0) >= UNIVERSE.min_market_cap]

    with open(MODELS_DIR / "latest_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    day["ml_score"] = bundle["booster"].predict(day[bundle["columns"]])

    comp_full = composite_score(feat[feat["date"] == asof])
    day["composite"] = comp_full.loc[day.index]

    day["ml_z"] = _zscore_by_date(day["ml_score"], day["date"])
    day["comp_z"] = _zscore_by_date(day["composite"], day["date"])
    b = COMPOSITE.ml_blend
    day["final_score"] = b * day["ml_z"] + (1 - b) * day["comp_z"]
    day["sector"] = day["ticker"].map(meta.set_index("ticker")["sector"])

    regime = load_regime()
    current_regime = regime.sort_values("date")["regime"].iloc[-1]
    print(f"market regime: {current_regime}")

    ranked = day.sort_values("final_score", ascending=False)
    out_cols = ["ticker", "sector", "close", "final_score", "ml_z", "comp_z",
                "ret_20", "rs_20", "hv_20", "market_cap"]
    out = ranked[out_cols].reset_index(drop=True)
    out.index += 1
    path = REPORTS_DIR / f"rank_{asof.date()}.csv"
    out.to_csv(path)
    print(f"wrote {path}")
    print("\nTop 40:")
    print(out.head(40).to_string())

    target = select_portfolio(ranked, current_regime)
    target = target.merge(day[["ticker", "sector", "close"]], on="ticker")
    print(f"\nTarget portfolio ({len(target)} names, "
          f"gross {target['weight'].sum():.0%}):")
    print(target.sort_values("weight", ascending=False).to_string(index=False))
