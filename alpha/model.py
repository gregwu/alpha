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

def _label_horizon(label: str) -> int:
    # "fwd_ret_20_rank" -> 20
    return int(label.split("_")[2])


def predictions_path(label: str):
    return DATA_DIR / f"predictions_{_label_horizon(label)}d.parquet"


def model_path(label: str):
    return MODELS_DIR / f"latest_model_{_label_horizon(label)}d.pkl"


def importance_path(label: str):
    return MODELS_DIR / f"feature_importance_{_label_horizon(label)}d.parquet"


def walk_forward_predict(feat: pd.DataFrame, start: str, end: str | None = None,
                         cfg=MODEL, label: str | None = None) -> pd.DataFrame:
    """Monthly-retrained walk-forward predictions of one label."""
    label = label or cfg.label
    cols = [c for c in all_feature_columns() if c in feat.columns]
    horizon = _label_horizon(label)
    gap = pd.Timedelta(days=cfg.embargo_days + int(horizon * 1.6))  # trading->calendar

    feat = feat.sort_values("date")
    dates = feat["date"]
    end_ts = pd.Timestamp(end) if end else dates.max()
    fold_starts = pd.date_range(pd.Timestamp(start), end_ts, freq=cfg.retrain_freq)

    trainable = feat["in_universe"] & feat[label].notna()
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
            lgb.Dataset(tr[cols], label=tr[label]),
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
            with open(model_path(label), "wb") as f:
                pickle.dump({"booster": booster, "columns": cols, "label": label,
                             "trained_through": str(train_end.date())}, f)

    result = pd.concat(preds, ignore_index=True)
    out_path = predictions_path(label)
    result.to_parquet(out_path, index=False)
    if importances:
        pd.concat(importances, axis=1).to_parquet(importance_path(label))
    log.info("wrote %s: %d prediction rows, %d folds", out_path, len(result), len(preds))
    return result


def load_blended_predictions(cfg=MODEL) -> pd.DataFrame:
    """Z-score-blend the per-horizon prediction files into one ml_score."""
    merged = None
    zcols = []
    for label, weight in cfg.labels.items():
        df = pd.read_parquet(predictions_path(label))
        col = f"ml_{_label_horizon(label)}"
        df = df.rename(columns={"ml_score": col})
        zcols.append((col, weight))
        merged = df if merged is None else merged.merge(df, on=["ticker", "date"], how="outer")

    total_w = sum(w for _, w in zcols)
    blend = 0.0
    any_valid = pd.Series(False, index=merged.index)
    for col, weight in zcols:
        by_date = merged.groupby("date")[col]
        z = (merged[col] - by_date.transform("mean")) / by_date.transform("std")
        blend = blend + (weight / total_w) * z.fillna(0.0)
        any_valid |= z.notna()
    merged["ml_score"] = blend.where(any_valid)
    return merged[["ticker", "date", "ml_score"]]
