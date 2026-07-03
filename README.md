# Alpha — Cross-Sectional Quant Equity System

Institutional-style ranking system per `prompt.txt`: many weak signals,
strict risk management, cross-sectional ranking instead of directional
prediction, walk-forward retraining, regime-gated exposure.

```text
Daily OHLCV (Postgres stock_data, 3,014 tickers, 2012+)
  -> Feature engineering (~100 features x 7 families)
  -> Market regime filter (bull/neutral/bear/high_vol/panic)
  -> LightGBM walk-forward ranking (monthly retrain, embargoed labels)
  -> Composite z-score blend
  -> Portfolio construction (top 30, sector caps, inverse-vol weights)
  -> Backtest / live ranking
```

## Quick start

```sh
./run.sh                  # refresh data + features + regime + live ranking
./run.sh --sync           # also update the DB via yahoostock sync_yahoo.py first
./run.sh --with-backtest  # additionally re-run the full walk-forward backtest
./start.sh                # web dashboard at http://localhost:8100
./stop.sh
```

## Web dashboard (`web/`)

FastAPI backend (`web/backend/main.py`) + React/Vite frontend
(`web/frontend/`). `start.sh` builds the frontend if needed and serves
everything from one uvicorn process on port 8100. Tabs: **Dashboard**
(equity/drawdown/exposure vs SPY), **Signals** (IC, decile spread, factor
IC by regime, feature importance), **Rankings** (live ranked list +
latest target portfolio), **Config** (read-only `alpha/config.py` view).
For frontend dev with hot reload: `cd web/frontend && npm run dev`
(proxies `/api` to :8100).

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Snapshot | `scripts/01_snapshot.py` | `data/prices.parquet`, `metadata`, `benchmark` (SPY/VIX via yfinance) |
| 2. Features | `scripts/02_features.py [workers]` | `data/features.parquet` (8.3M rows x 108 cols) |
| 3. Research run | `scripts/03_backtest.py [--skip-ml]` | predictions, backtest, `reports/backtest_report.txt`, charts |
| 4. Live ranking | `scripts/04_rank_today.py` | `reports/rank_<date>.csv` + target portfolio |
| 5. Report export | `scripts/05_export_report.py` | `reports/report_data.json` for the dashboard |

DB credentials come from `.env` (same Postgres as pattern_scan, port 5433).

## Design choices & guardrails

- **Point-in-time discipline.** Every feature uses trailing windows only;
  swing points confirm K bars late; labels (`fwd_ret_{5,20}` + their daily
  cross-sectional rank) are never available to the model within the
  embargo (21 trading days + horizon) before each fold's first prediction.
- **Universe** (`alpha/data.py`): price >= $10, 63-day avg dollar volume >=
  $20M, >=300 days of history, sector known (filters out ETFs/funds).
  Market-cap >= $1B is applied **only live** — the metadata snapshot is
  current-day, so using it historically would inject survivorship bias.
- **Ranking, not prediction** (`alpha/model.py`): label = percentile rank
  of 20d forward return within that day's universe. Monthly retrain on a
  4-year trailing window.
- **Composite** (`alpha/composite.py`): winsorized cross-sectional
  z-scores per factor family, spec weights (RS 25 / trend+momentum 20 /
  volume 15 / vol 10 / structure 10). Fundamentals & options weights are
  redistributed until those data sources are added. Final score =
  50% ML z + 50% composite z.
- **Regime** (`alpha/regime.py`): rule-based from SPY 200dma, drawdown,
  VIX, realized vol, universe breadth; 3-day debounce, panic overrides.
  Regime scales gross exposure (bull 1.0 ... panic 0.1) — it never picks
  stocks.
- **Portfolio** (`alpha/portfolio.py`): top 30 by final score, max 6 per
  sector (20%), inverse-vol weights capped at 5%, regime-scaled gross.
  Weekly rebalance, next-close execution, 10bps one-way costs.
- **Turnover control** (`rebalance_portfolio`): asymmetric entry/exit —
  buy only into the top 30, but hold until a name decays below rank 105
  (top_n x keep_buffer 3.5); weight adjustments under 1% are skipped.
  Sweep result: one-way turnover ~54x -> ~34x/yr, ~+0.05 Sharpe vs
  reconstructing the portfolio fresh every week.
- **Graph layer** (`alpha/features/graph.py`): monthly stock graph from
  trailing 126d return correlations of the then-current universe
  (top-10 positively correlated peers, correlation-weighted). Daily
  features: neighbor momentum (ret 20/60, RS 20), own-vs-neighbor
  momentum gap (lead-lag), centrality, average correlation. Feeds the
  ML model and attribution, not the fixed-weight composite.

## Feature families (`alpha/features/`)

trend (EMA/SMA distances+slopes, regression slope/acceleration, ADX),
momentum (multi-horizon returns, 12-1, RSI, consistency, 52wk-high
proximity), relative strength (vs SPY at 4 horizons, RS-line slope, beta,
vs sector), volume (rel volume, OBV slope, VWAP distance, CMF, up/down
volume), volatility (ATR, HV, Parkinson, BB width percentile,
contraction/expansion, skew), ICT structure (BOS counts, FVG counts &
size, liquidity sweeps, order-block distance, premium/discount in 60d
range), sector (sector RS, breadth, momentum rank), plus daily
cross-sectional percentile ranks of key signals.

## Extending (the spec's roadmap)

- **Fundamentals / options factor groups**: plug into
  `FEATURE_GROUPS` + composite weights; needs point-in-time sources
  (yfinance snapshots are current-only — fine live, leaky in backtests).
- **Richer graph edges**: supply-chain links and common institutional
  ownership alongside the existing correlation edges.
- **Monthly feedback loop**: rerun 02+03; `models/feature_importance.parquet`
  and the factor-IC-by-regime table in the report show which families are
  earning their place.
