"""Recompute report diagnostics (IC, deciles, attribution) from existing
artifacts and export reports/report_data.json for the web dashboard.

Useful when the backtest artifacts predate the JSON export step.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd

from alpha.backtest import build_scores
from alpha.composite import composite_score, group_score_frame
from alpha.data import load_metadata
from alpha.features.build import FEATURES_PATH
from alpha.model import load_blended_predictions
from alpha.regime import load_regime
from alpha.report import (decile_returns, export_report_data,
                          factor_ic_by_regime, information_coefficient)

if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    meta = load_metadata()
    preds = load_blended_predictions()
    regime = load_regime()

    comp = composite_score(feat)
    feat["sector"] = feat["ticker"].map(meta.set_index("ticker")["sector"])
    scores = build_scores(feat, preds, comp)

    ic = information_coefficient(scores, feat)
    dec = decile_returns(scores, feat)
    groups = group_score_frame(feat)
    attribution = factor_ic_by_regime(feat, groups, regime)
    export_report_data(ic, dec, attribution)
    print("wrote reports/report_data.json")
