"""Walk-forward cross-sectional ranking model.

Asks "which stocks will outperform the others over the next 20 days",
not "will this stock go up": the label is each stock's percentile rank
of forward return within that day's universe.

Protocol per fold (monthly):
  train window  = `train_years` of history ending `embargo_days` +
                  label horizon before the first prediction date
  predict range = the following month

Only in-universe rows with non-NaN labels are trained on. Predictions are
written for every in-universe row of the prediction month.
"""

import logging
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DATA_DIR, MODEL, MODELS_DIR
from .features.build import all_feature_columns

log = logging.getLogger(__name__)

PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"


def _label_horizon(label: str) -> int:
    # "fwd_ret_20_rank" -> 20
    return int(label.split("_")[2])


def walk_forward_predict(feat: pd.DataFrame, start: str, end: str | None = None,
                         cfg=MODEL) -> pd.DataFrame:
    """Monthly-retrained walk-forward predictions of `cfg.label`."""
    cols = [c for c in all_feature_columns() if c in feat.columns]
    horizon = _label_horizon(cfg.label)
    gap = pd.Timedelta(days=cfg.embargo_days + int(horizon * 1.6))  # trading->calendar

    feat = feat.sort_values("date")
    dates = feat["date"]
    end_ts = pd.Timestamp(end) if end else dates.max()
    fold_starts = pd.date_range(pd.Timestamp(start), end_ts, freq=cfg.retrain_freq)

    trainable = feat["in_universe"] & feat[cfg.label].notna()
    preds, importances = [], []

    for i, fs in enumerate(fold_starts):
        fe = fold_starts[i + 1] if i + 1 < len(fold_starts) else end_ts + pd.Timedelta(days=1)
        train_end = fs - gap
        train_start = train_end - pd.DateOffset(years=cfg.train_years)

        tr = feat[trainable & (dates >= train_start) & (dates < train_end)]
        if len(tr) < 50_000:
            log.warning("fold %s: only %d train rows, skipping", fs.date(), len(tr))
            continue
        if len(tr) > cfg.max_train_rows:
            tr = tr.sample(cfg.max_train_rows, random_state=42)

        booster = lgb.train(
            cfg.lgb_params,
            lgb.Dataset(tr[cols], label=tr[cfg.label]),
            num_boost_round=cfg.num_boost_round,
        )

        pr = feat[feat["in_universe"] & (dates >= fs) & (dates < fe)]
        if pr.empty:
            continue
        out = pr[["ticker", "date"]].copy()
        out["ml_score"] = booster.predict(pr[cols])
        preds.append(out)

        imp = pd.Series(booster.feature_importance("gain"), index=cols)
        importances.append(imp.rename(fs))
        log.info("fold %s: trained on %d rows (%s..%s), predicted %d rows",
                 fs.date(), len(tr), train_start.date(), train_end.date(), len(out))

        if i == len(fold_starts) - 1:
            with open(MODELS_DIR / "latest_model.pkl", "wb") as f:
                pickle.dump({"booster": booster, "columns": cols, "label": cfg.label,
                             "trained_through": str(train_end.date())}, f)

    result = pd.concat(preds, ignore_index=True)
    result.to_parquet(PREDICTIONS_PATH, index=False)
    if importances:
        pd.concat(importances, axis=1).to_parquet(MODELS_DIR / "feature_importance.parquet")
    log.info("wrote %s: %d prediction rows, %d folds", PREDICTIONS_PATH, len(result), len(preds))
    return result


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PREDICTIONS_PATH)
