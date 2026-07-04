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
from alpha.config import COMPOSITE, MODEL, REPORTS_DIR, UNIVERSE
from alpha.model import model_path
from alpha.data import load_benchmark, load_metadata
from alpha.features.build import FEATURES_PATH
from alpha.portfolio import select_portfolio
from alpha.regime import load_regime

MAX_STALE_BDAYS = 3   # refuse to rank on data older than this

if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    meta = load_metadata()
    asof = feat["date"].max()

    stale_bdays = int(np.busday_count(asof.date(), pd.Timestamp.now().date()))
    if stale_bdays > MAX_STALE_BDAYS and "--allow-stale" not in sys.argv:
        print(f"ERROR: feature data is {stale_bdays} business days old "
              f"(as-of {asof.date()}). The sync likely failed — fix it or "
              f"re-run with --allow-stale to rank anyway.")
        sys.exit(2)

    day = feat[(feat["date"] == asof) & feat["in_universe"]].copy()
    print(f"as-of date: {asof.date()}  universe: {len(day)}")

    # Live-only market-cap filter
    mcap = meta.set_index("ticker")["market_cap"]
    day["market_cap"] = day["ticker"].map(mcap)
    day = day[day["market_cap"].fillna(0) >= UNIVERSE.min_market_cap]

    # Blend the per-horizon models' z-scores (same scheme as the backtest)
    total_w = sum(MODEL.labels.values())
    blend = 0.0
    for label, weight in MODEL.labels.items():
        with open(model_path(label), "rb") as f:
            bundle = pickle.load(f)
        raw = pd.Series(bundle["booster"].predict(day[bundle["columns"]]), index=day.index)
        z = (raw - raw.mean()) / raw.std()
        blend = blend + (weight / total_w) * z
    day["ml_score"] = blend

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

    # Enrich the top names with LIVE EDGAR fundamentals (real-time per-ticker
    # API — fresher than the monthly bulk archive; cached + SEC-throttled).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trading"))
        import edgar_fundamentals as E

        cik = E.ticker_to_cik()
        top = out.head(60)
        live = {}
        for t, px in zip(top["ticker"], top["close"]):
            f = E.get_fundamentals(t.replace(".US", ""), cik, price=float(px))
            if not f.get("error"):
                live[t] = f
        for col in ("pe", "eps_yoy", "revenue_yoy", "roe", "debt_to_equity"):
            out[f"live_{col}"] = out["ticker"].map(
                lambda t: live.get(t, {}).get(col))
        print(f"live EDGAR fundamentals: {len(live)}/{len(top)} top names enriched")
    except Exception as e:  # noqa: BLE001 — enrichment must never block ranking
        print(f"live fundamentals enrichment skipped: {e}")
    path = REPORTS_DIR / f"rank_{asof.date()}.csv"
    out.to_csv(path)
    print(f"wrote {path}")
    print("\nTop 40:")
    print(out.head(40).to_string())

    # Turnover-aware target: rebalance from the previously published target
    # (same rank-buffer logic as the backtest) instead of rebuilding fresh.
    import json

    target_path = REPORTS_DIR / "target_portfolio.json"
    current = pd.Series(dtype=float)
    if target_path.exists():
        prev = json.loads(target_path.read_text())
        current = pd.Series(prev.get("weights", {}), dtype=float)
        current.index = [t if t.endswith(".US") else t + ".US" for t in current.index]

    if len(current):
        from alpha.portfolio import rebalance_portfolio
        weights = rebalance_portfolio(ranked, current_regime, current)
    else:
        weights = select_portfolio(ranked, current_regime).set_index("ticker")["weight"]

    target = weights.rename("weight").reset_index().rename(columns={"index": "ticker"})
    target = target.merge(day[["ticker", "sector", "close"]], on="ticker", how="left")
    print(f"\nTarget portfolio ({len(target)} names, "
          f"gross {target['weight'].sum():.0%}):")
    print(target.sort_values("weight", ascending=False).to_string(index=False))

    target_path.write_text(json.dumps({
        "asof": str(asof.date()),
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "regime": current_regime,
        "gross": round(float(target["weight"].sum()), 4),
        "weights": {t.replace(".US", ""): round(float(w), 5)
                    for t, w in weights.items()},
    }, indent=1))
    print(f"wrote {target_path}")
