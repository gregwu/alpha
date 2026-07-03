"""Train the latest (live) booster for each configured label horizon on
the most recent training window. Fast (~30s) — suitable for the daily
pipeline so live rankings always use a current model, without re-running
the full walk-forward backtest.
"""

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_latest")

import lightgbm as lgb
import pandas as pd

from alpha.config import MODEL
from alpha.features.build import FEATURES_PATH, all_feature_columns
from alpha.model import _label_horizon, model_path

if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    cols = [c for c in all_feature_columns() if c in feat.columns]

    for label in MODEL.labels:
        horizon = _label_horizon(label)
        gap = pd.Timedelta(days=MODEL.embargo_days + int(horizon * 1.6))
        train_end = feat["date"].max() - gap
        train_start = train_end - pd.DateOffset(years=MODEL.train_years)
        tr = feat[feat["in_universe"] & feat[label].notna()
                  & (feat["date"] >= train_start) & (feat["date"] < train_end)]
        if len(tr) > MODEL.max_train_rows:
            tr = tr.sample(MODEL.max_train_rows, random_state=42)

        booster = lgb.train(MODEL.lgb_params, lgb.Dataset(tr[cols], label=tr[label]),
                            num_boost_round=MODEL.num_boost_round)
        with open(model_path(label), "wb") as f:
            pickle.dump({"booster": booster, "columns": cols, "label": label,
                         "trained_through": str(train_end.date())}, f)
        log.info("%s: trained on %d rows through %s -> %s",
                 label, len(tr), train_end.date(), model_path(label).name)
