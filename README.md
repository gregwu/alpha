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
./run.sh                  # daily: data + features + regime + live boosters + ranking
./run.sh --sync           # also update the DB via yahoostock sync_yahoo.py first
./run.sh --fundamentals   # force EDGAR companyfacts refresh (else auto-monthly)
./run.sh --with-backtest  # force the full walk-forward backtest (else auto-monthly)
./start.sh                # web dashboard at http://localhost:8100
./stop.sh
./scripts/install_launchd.sh   # fully hands-off: pipeline weekdays 18:30 +
                               # dashboard keepalive at login (auto-restart)
```

Execution (`trading/` — vendored from tradingview-mcp):
`trading_service.py` is the only process that talks to Alpaca (Flask on
:8787, token auth, PAPER unless `trading/.env` sets `ALPACA_LIVE=1`;
launchd keepalive `com.gangwu.alpha.trading`). `scripts/09_alpha_bot.py`
publishes the pipeline's weekly target through it — local paper book by
default, `--live-trade` routes orders, `--live-ok` additionally required
for a LIVE service. Weekly-idempotent, stale-target guard, position/gross
ceilings, history in `reports/alpha_bot_trades.csv`. `./run.sh --trade`
chains it after the daily ranking. The three reference bots
(momentum / lev_trend / rsi_pivot) and `run_bot.sh` are vendored
alongside; Alpaca keys live in `trading/.env` (gitignored).

Automated housekeeping: EDGAR fundamentals re-download when >25 days old;
full walk-forward retrain when predictions >28 days old; live rankings
refuse data >3 business days stale (`--allow-stale` overrides); every
research run appends to `reports/metrics_history.csv` (edge-drift audit
trail); logs and daily rank CSVs pruned after 90 days.

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
- **Ranking, not prediction** (`alpha/model.py`): two LightGBM models,
  labels = percentile rank of 5d and 20d forward return within that
  day's universe, blended 50/50 by daily z-score. The 5d model has the
  higher IC at *both* horizons (faster label = more independent training
  examples). Monthly retrain on a 4-year trailing window.
  Blended IC +0.023 vs fwd 20d; decile spread D1 +0.71% -> D10 +1.17%.
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

## Current results (walk-forward 2016 -> 2026-07, 10bps costs + financing)

Goal: beat SPY — currently met. Config: top-20, 5d+20d ML blend,
PIT EDGAR fundamentals, bull-regime leverage 1.25x (financing charged
at T-bill + 1%). **CAGR 16.4% vs SPY 15.0% (excess +1.4%, info ratio
+0.19, alpha +1.1%/yr, beta 1.18)**, Sharpe 0.67 vs 0.87, MaxDD -46%
vs -34%, turnover 50x one-way. The excess return is paid for with
higher vol and deeper drawdowns; a drawdown-managed profile (top-30 +
defensive regime ladder, see `PortfolioConfig.regime_exposure` comment)
runs MaxDD ~-27% at ~9% CAGR. Adding fundamentals was what flipped the
excess return positive (25% of 20d-model gain).

## Tested & rejected (2026-07, don't re-try blindly)

- **Correlation-penalized sizing** (`corr_penalty`, kept in code, default
  0): weight ~ 1/(vol x avg_corr) made things monotonically worse
  (Sharpe 0.54 -> 0.48, MaxDD -39% -> -45% as penalty rose to 1.5) —
  penalizing correlated names tilts the book into idiosyncratic
  small-caps whose blowup risk dominates the cluster risk removed.
- **Retuning the regime exposure ladder**: both a more defensive ladder
  (1/.8/.4/.15/0) and full-neutral (1/1/.5/.3/.1) underperformed the
  baseline (1/.7/.5/.3/.1). An always-100% control scored the *same
  Sharpe* (0.53 vs 0.54) with CAGR 9.8% vs 7.4% and MaxDD -48% vs -39%:
  the regime layer is pure risk scaling, not alpha timing — keep it for
  drawdown control, don't expect it to add return.

## Extending (the spec's roadmap)

- **Fundamentals**: DONE — point-in-time from SEC EDGAR companyfacts
  (`scripts/08_fundamentals.py`, features available on *filing* date).
  Refresh the bulk zip + re-run monthly.
- **Options flow**: dropped — yfinance options data is unreliable (no
  history, stale OI); would need a paid source (ORATS/CBOE) to do
  honestly. The 5% composite weight stays redistributed.
- **Richer graph edges**: supply-chain links and common institutional
  ownership alongside the existing correlation edges.
- **Monthly feedback loop**: rerun 02+03; per-horizon
  `models/feature_importance_{5,20}d.parquet` and the
  factor-IC-by-regime table show which families are earning their place.
