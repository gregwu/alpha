"""FastAPI backend for the alpha dashboard.

Read-only API over the research artifacts in data/, models/, reports/.
Run:  uvicorn web.backend.main:app --reload --port 8100
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from alpha import config as cfg
from alpha.report import perf_metrics, relative_metrics

app = FastAPI(title="alpha dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA = cfg.DATA_DIR
MODELS = cfg.MODELS_DIR
REPORTS = cfg.REPORTS_DIR


def _clean(obj):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (obj is None or math.isnan(obj) or math.isinf(obj)) else float(obj)
    return obj


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found — run {hint} first")
    return path


@app.get("/api/config")
def get_config():
    out = {}
    for name in ("UNIVERSE", "FEATURES", "MODEL", "PORTFOLIO", "BACKTEST", "COMPOSITE"):
        out[name.lower()] = dataclasses.asdict(getattr(cfg, name))
    return _clean(out)


@app.get("/api/status")
def get_status():
    out = {}
    for label, path in [("prices", DATA / "prices.parquet"),
                        ("features", DATA / "features.parquet"),
                        ("predictions_20d", DATA / "predictions_20d.parquet"),
                        ("predictions_5d", DATA / "predictions_5d.parquet"),
                        ("benchmark", DATA / "benchmark.parquet")]:
        if path.exists():
            dates = pd.read_parquet(path, columns=["date"])["date"]
            out[label] = {"max_date": str(dates.max().date()), "rows": int(len(dates))}
        else:
            out[label] = None
    reg_path = DATA / "regime.parquet"
    if reg_path.exists():
        reg = pd.read_parquet(reg_path).sort_values("date")
        last = reg.iloc[-1]
        out["regime"] = {"current": last["regime"], "as_of": str(last["date"].date()),
                         "vix": _clean(float(last["vix"])),
                         "pct_above_200": _clean(float(last["pct_above_200"]))}
    import pickle
    models = []
    for pkl in sorted(MODELS.glob("latest_model_*.pkl")):
        with open(pkl, "rb") as f:
            b = pickle.load(f)
        models.append({"label": b["label"], "trained_through": b["trained_through"],
                       "n_features": len(b["columns"])})
    if models:
        out["model"] = models[0]      # back-compat for the header
        out["models"] = models
    return out


@app.get("/api/backtest/summary")
def backtest_summary():
    path = _need(DATA / "backtest_daily.parquet", "scripts/03_backtest.py")
    daily = pd.read_parquet(path).set_index("date")
    return _clean({
        "period": {"start": str(daily.index.min().date()), "end": str(daily.index.max().date())},
        "strategy": perf_metrics(daily, "ret"),
        "spy": perf_metrics(daily, "spy_ret"),
        "vs_spy": relative_metrics(daily),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
        "annual_turnover": float(daily["cost"].sum() / (cfg.PORTFOLIO.cost_bps / 1e4)
                                 / (len(daily) / 252)),
    })


@app.get("/api/backtest/series")
def backtest_series():
    path = _need(DATA / "backtest_daily.parquet", "scripts/03_backtest.py")
    daily = pd.read_parquet(path)
    reg = pd.read_parquet(DATA / "regime.parquet")[["date", "regime"]] \
        if (DATA / "regime.parquet").exists() else None
    if reg is not None:
        daily = daily.merge(reg, on="date", how="left")
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    daily["spy_drawdown"] = daily["spy_equity"] / daily["spy_equity"].cummax() - 1
    cols = ["date", "equity", "spy_equity", "drawdown", "spy_drawdown",
            "gross_exposure", "regime"]
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return _clean(daily[[c for c in cols if c in daily.columns]].to_dict(orient="records"))


@app.get("/api/backtest/holdings")
def backtest_holdings():
    path = _need(DATA / "backtest_holdings.parquet", "scripts/03_backtest.py")
    h = pd.read_parquet(path)
    last_date = h["date"].max()
    latest = h[h["date"] == last_date].sort_values("weight", ascending=False)
    meta = pd.read_parquet(DATA / "metadata.parquet")
    latest = latest.merge(meta[["ticker", "sector"]], on="ticker", how="left")
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    return _clean(latest.to_dict(orient="records"))


@app.get("/api/report")
def get_report():
    """Text report plus structured diagnostics (deciles, IC, attribution)."""
    out = {"text": None, "data": None}
    txt = REPORTS / "backtest_report.txt"
    if txt.exists():
        out["text"] = txt.read_text()
    js = REPORTS / "report_data.json"
    if js.exists():
        out["data"] = json.loads(js.read_text())
    return out


@app.get("/api/importance")
def feature_importance(top: int = 25, horizon: int = 20):
    path = _need(MODELS / f"feature_importance_{horizon}d.parquet", "scripts/03_backtest.py")
    imp = pd.read_parquet(path)
    mean_imp = imp.mean(axis=1).sort_values(ascending=False)
    total = mean_imp.sum()
    return _clean([
        {"feature": k, "gain_share": float(v / total)}
        for k, v in mean_imp.head(top).items()
    ])


@app.get("/api/rankings")
def rankings(limit: int = 100):
    files = sorted(REPORTS.glob("rank_*.csv"))
    if not files:
        raise HTTPException(404, "no rank_*.csv yet — run scripts/04_rank_today.py")
    latest = files[-1]
    df = pd.read_csv(latest, index_col=0)
    return _clean({
        "as_of": latest.stem.replace("rank_", ""),
        "rows": df.head(limit).reset_index().rename(columns={"index": "rank"})
                  .to_dict(orient="records"),
    })


@app.get("/api/regime/series")
def regime_series():
    path = _need(DATA / "regime.parquet", "scripts/03_backtest.py")
    reg = pd.read_parquet(path)
    reg["date"] = reg["date"].dt.strftime("%Y-%m-%d")
    cols = ["date", "regime", "spy_close", "vix", "spy_dd", "pct_above_200"]
    return _clean(reg[cols].to_dict(orient="records"))


# Serve the built frontend when web/frontend/dist exists (production mode)
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
