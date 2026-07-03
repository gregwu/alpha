"""End-to-end research run: regime -> composite -> walk-forward ML ->
portfolio backtest -> report.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run")

import pandas as pd

from alpha.backtest import build_scores, run_backtest
from alpha.composite import composite_score, group_score_frame
from alpha.config import BACKTEST, DATA_DIR
from alpha.data import load_benchmark, load_metadata, load_prices
from alpha.features.build import FEATURES_PATH
from alpha.config import MODEL
from alpha.model import walk_forward_predict, load_blended_predictions
from alpha.regime import compute_breadth, classify_regime
from alpha.report import (decile_returns, factor_ic_by_regime,
                          information_coefficient, write_report)

SKIP_ML = "--skip-ml" in sys.argv

if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    bench = load_benchmark()
    meta = load_metadata()
    log.info("features loaded: %s", feat.shape)

    # Regime
    breadth = compute_breadth(feat)
    regime = classify_regime(bench, breadth)
    log.info("regime distribution:\n%s", regime["regime"].value_counts().to_string())

    # Composite + factor-group scores
    comp = composite_score(feat)
    comp.rename("composite").to_frame().assign(
        ticker=feat["ticker"], date=feat["date"]
    ).dropna().to_parquet(DATA_DIR / "composite.parquet", index=False)
    log.info("composite done")

    # Walk-forward ML: one model per label horizon, blended by z-score
    if not SKIP_ML:
        for label in MODEL.labels:
            walk_forward_predict(feat, start=BACKTEST.start, label=label)
    preds = load_blended_predictions()

    # Blend, construct, backtest
    feat["sector"] = feat["ticker"].map(meta.set_index("ticker")["sector"])
    scores = build_scores(feat, preds, comp)
    daily = run_backtest(scores, load_prices(), regime, bench)

    # Research diagnostics
    ic = information_coefficient(scores, feat)
    dec = decile_returns(scores, feat)
    groups = group_score_frame(feat)
    attribution = factor_ic_by_regime(feat, groups, regime)
    holdings = pd.read_parquet(DATA_DIR / "backtest_holdings.parquet")
    text = write_report(daily.copy(), ic, dec, attribution, holdings)
    print(text)
