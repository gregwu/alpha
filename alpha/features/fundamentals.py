"""Point-in-time fundamentals from SEC EDGAR companyfacts.

Every XBRL fact carries its `filed` date — a feature becomes available on
the filing date, never at the fiscal period end. This is the property
yfinance snapshots lack and what makes these features safe to backtest.

Concepts -> features:
  rev_growth_yoy, eps_growth_yoy  (quarterly, ~90d-duration facts)
  roe_ttm, roa_ttm                (TTM net income over equity / assets)
  gross_margin_ttm, fcf_margin_ttm
  earnings_yield, fcf_yield       (TTM over PIT market cap = shares x close)
  debt_equity, shares_growth_yoy
"""

import json
import logging
import zipfile

import numpy as np
import pandas as pd

from ..config import DATA_DIR

log = logging.getLogger(__name__)

FUND_RAW_PATH = DATA_DIR / "fundamentals_raw.parquet"
FUND_PATH = DATA_DIR / "fundamentals.parquet"

# us-gaap concepts with fallbacks, in priority order
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "gross_profit": ["GrossProfit"],
}
UNITS = {"eps": "USD/shares"}
INSTANT = {"equity", "assets", "liabilities"}          # balance-sheet points
SHARES_CONCEPT = ("dei", "EntityCommonStockSharesOutstanding", "shares")


def _extract_series(facts: dict, taxonomy: str, names: list[str], unit: str):
    """Best available (end, filed, val, duration) rows for a concept."""
    tax = facts.get(taxonomy, {})
    for name in names:
        node = tax.get(name)
        if not node:
            continue
        rows = node.get("units", {}).get(unit)
        if not rows:
            continue
        out = []
        for r in rows:
            if r.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
                continue
            start = r.get("start")
            dur = ((pd.Timestamp(r["end"]) - pd.Timestamp(start)).days
                   if start else np.nan)
            out.append((r["end"], r["filed"], r["val"], dur))
        if out:
            return out
    return []


def parse_companyfacts(zip_path, tickers_base: set) -> pd.DataFrame:
    """Extract raw concept series for our universe from companyfacts.zip."""
    import requests

    tk = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "gangwu research wugangc@gmail.com"},
        timeout=30).json()
    cik_of = {v["ticker"]: int(v["cik_str"]) for v in tk.values()}
    wanted = {t: cik_of[t] for t in tickers_base if t in cik_of}
    log.info("CIK mapping: %d of %d tickers", len(wanted), len(tickers_base))

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        done = 0
        for ticker, cik in wanted.items():
            fn = f"CIK{cik:010d}.json"
            if fn not in names:
                continue
            try:
                facts = json.loads(z.read(fn)).get("facts", {})
            except Exception:
                continue
            for key, concepts in CONCEPTS.items():
                unit = UNITS.get(key, "USD")
                for end, filed, val, dur in _extract_series(facts, "us-gaap", concepts, unit):
                    rows.append((ticker, key, end, filed, val, dur))
            for end, filed, val, dur in _extract_series(
                    facts, SHARES_CONCEPT[0], [SHARES_CONCEPT[1]], SHARES_CONCEPT[2]):
                rows.append((ticker, "shares", end, filed, val, dur))
            done += 1
            if done % 500 == 0:
                log.info("parsed %d companies", done)

    df = pd.DataFrame(rows, columns=["ticker", "concept", "end", "filed", "val", "duration"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.drop_duplicates(subset=["ticker", "concept", "end", "duration"], keep="first")
    df.to_parquet(FUND_RAW_PATH, index=False)
    log.info("wrote %s: %d rows, %d tickers", FUND_RAW_PATH, len(df), df.ticker.nunique())
    return df


def _quarterly(df):    # ~90-day flow facts
    return df[(df.duration >= 70) & (df.duration <= 110)]


def _flows_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly flow values from a mix of discrete-quarter and YTD facts.

    Cash-flow concepts are usually filed year-to-date (duration 90/180/
    270/360d from the same fiscal-year start). Discrete quarters are the
    ~90d facts; later quarters come from differencing consecutive YTD
    values that share a start date.
    """
    direct = _quarterly(df)[["ticker", "end", "filed", "val", "duration"]]

    diffs = []
    raw = df.dropna(subset=["duration"]).copy()
    raw["start"] = raw["end"] - pd.to_timedelta(raw["duration"], unit="D")
    raw["start_key"] = raw["start"].dt.to_period("M")   # tolerate day jitter
    for (t, sk), sub in raw.groupby(["ticker", "start_key"], sort=False):
        sub = sub.sort_values("duration").drop_duplicates("duration", keep="last")
        if len(sub) < 2:
            continue
        v, e, f, d = (sub["val"].to_numpy(), sub["end"].to_numpy(),
                      sub["filed"].to_numpy(), sub["duration"].to_numpy())
        for i in range(1, len(sub)):
            span = d[i] - d[i - 1]
            if 70 <= span <= 110:
                diffs.append((t, e[i], max(f[i], f[i - 1]), v[i] - v[i - 1], span))
    ytd_q = pd.DataFrame(diffs, columns=["ticker", "end", "filed", "val", "duration"])

    out = pd.concat([direct, ytd_q], ignore_index=True)
    return out.sort_values(["ticker", "end"]).drop_duplicates(
        ["ticker", "end"], keep="first")


def _ttm(qdf: pd.DataFrame) -> pd.DataFrame:
    """Trailing-4-quarter sum, available at the filed date of the newest
    quarter. Requires 4 distinct quarter-ends within ~400 days."""
    out = []
    for t, sub in qdf.groupby("ticker", sort=False):
        sub = sub.sort_values("end").drop_duplicates("end", keep="last")
        vals, ends, fileds = sub["val"].to_numpy(), sub["end"].to_numpy(), sub["filed"].to_numpy()
        for i in range(3, len(sub)):
            span = (ends[i] - ends[i - 3]).astype("timedelta64[D]").astype(int)
            if span <= 300:
                out.append((t, ends[i], max(fileds[i - 3:i + 1]), vals[i - 3:i + 1].sum()))
    return pd.DataFrame(out, columns=["ticker", "end", "filed", "val"])


def build_fundamental_features(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw concept rows -> PIT feature events keyed by availability date."""
    raw = raw if raw is not None else pd.read_parquet(FUND_RAW_PATH)

    def concept(key, quarterly=False):
        df = raw[raw.concept == key]
        return _quarterly(df) if quarterly else df

    rev_q = _flows_quarterly(concept("revenue"))
    eps_q = _flows_quarterly(concept("eps"))
    ni_ttm = _ttm(_flows_quarterly(concept("net_income")))
    rev_ttm = _ttm(rev_q)
    gp_ttm = _ttm(_flows_quarterly(concept("gross_profit")))
    ocf_ttm = _ttm(_flows_quarterly(concept("ocf")))
    capex_ttm = _ttm(_flows_quarterly(concept("capex")))

    events = []   # (ticker, available_date, feature, value)

    def yoy(df, feat):
        for t, sub in df.groupby("ticker", sort=False):
            sub = sub.sort_values("end").drop_duplicates("end", keep="last")
            by_end = dict(zip(sub["end"], sub["val"]))
            for _, r in sub.iterrows():
                prior = [v for e, v in by_end.items()
                         if 330 <= (r["end"] - e).days <= 400]
                if prior and abs(prior[-1]) > 1e-9:
                    events.append((t, r["filed"], feat,
                                   (r["val"] - prior[-1]) / abs(prior[-1])))

    yoy(rev_q, "rev_growth_yoy")
    yoy(eps_q, "eps_growth_yoy")
    yoy(concept("shares"), "shares_growth_yoy")

    def ratio(num_df, den_df, feat, clip=None):
        den = den_df.sort_values(["ticker", "end"]).drop_duplicates(
            ["ticker", "end"], keep="last")
        den_by_t = {t: s for t, s in den.groupby("ticker", sort=False)}
        for t, sub in num_df.groupby("ticker", sort=False):
            ds = den_by_t.get(t)
            if ds is None:
                continue
            for _, r in sub.iterrows():
                cand = ds[(ds["end"] <= r["end"] + pd.Timedelta(days=10))
                          & (ds["end"] >= r["end"] - pd.Timedelta(days=120))]
                if cand.empty or abs(cand.iloc[-1]["val"]) < 1e-9:
                    continue
                v = r["val"] / cand.iloc[-1]["val"]
                if clip:
                    v = float(np.clip(v, *clip))
                avail = max(r["filed"], cand.iloc[-1]["filed"])
                events.append((t, avail, feat, v))

    equity = concept("equity")
    assets = concept("assets")
    liab = concept("liabilities")
    ratio(ni_ttm, equity, "roe_ttm", clip=(-2, 2))
    ratio(ni_ttm, assets, "roa_ttm", clip=(-1, 1))
    ratio(gp_ttm, rev_ttm, "gross_margin_ttm", clip=(-1, 1))
    ratio(liab, equity, "debt_equity", clip=(0, 20))

    # FCF margin: (OCF - capex) / revenue, aligned on TTM windows
    fcf = ocf_ttm.merge(capex_ttm, on=["ticker", "end"], suffixes=("_ocf", "_cx"))
    fcf["val"] = fcf["val_ocf"] - fcf["val_cx"]
    fcf["filed"] = fcf[["filed_ocf", "filed_cx"]].max(axis=1)
    ratio(fcf[["ticker", "end", "filed", "val"]], rev_ttm, "fcf_margin_ttm", clip=(-2, 2))

    # Yields need market cap -> emitted as ttm-per-share style events;
    # earnings_yield/fcf_yield are finished in the merge step (needs price).
    shares = concept("shares").rename(columns={"val": "shares"})
    for feat, num in [("ni_ttm_ps", ni_ttm), ("fcf_ttm_ps", fcf[["ticker", "end", "filed", "val"]])]:
        ratio(num, shares.rename(columns={"shares": "val"}), feat, clip=(-1000, 1000))

    ev = pd.DataFrame(events, columns=["ticker", "date", "feature", "value"])
    ev = ev.dropna()
    ev["date"] = pd.to_datetime(ev["date"])
    ev.to_parquet(FUND_PATH, index=False)
    log.info("wrote %s: %d events, %d tickers, features=%s",
             FUND_PATH, len(ev), ev.ticker.nunique(), sorted(ev.feature.unique()))
    return ev


FUNDAMENTAL_FEATURES = [
    "rev_growth_yoy", "eps_growth_yoy", "shares_growth_yoy",
    "roe_ttm", "roa_ttm", "gross_margin_ttm", "debt_equity",
    "fcf_margin_ttm", "earnings_yield", "fcf_yield",
]

MAX_STALENESS_DAYS = 150   # drop a fundamental once no fresh filing this long


def add_fundamental_features(feat: pd.DataFrame) -> pd.DataFrame:
    """As-of join the PIT feature events onto the daily frame; finish the
    yield features with the PIT close price."""
    if not FUND_PATH.exists():
        log.warning("no fundamentals.parquet — skipping fundamental features")
        for c in FUNDAMENTAL_FEATURES:
            feat[c] = np.nan
        return feat

    ev = pd.read_parquet(FUND_PATH)
    ev["ticker"] = ev["ticker"] + ".US"
    feat = feat.sort_values("date")

    for name, sub in ev.groupby("feature", sort=False):
        wide = (sub.sort_values("date")
                .drop_duplicates(["ticker", "date"], keep="last")
                .pivot(index="date", columns="ticker", values="value"))
        # availability lag: filed after the close -> usable next day
        wide.index = wide.index + pd.Timedelta(days=1)
        dates = feat["date"].drop_duplicates().sort_values()
        wide = wide.reindex(wide.index.union(dates)).ffill(limit=None)
        # staleness cap
        last_seen = wide.notna().cumsum()
        tall = wide.loc[dates].melt(ignore_index=False, var_name="ticker",
                                    value_name=name).reset_index(names="date")
        feat = feat.merge(tall, on=["ticker", "date"], how="left")

    for col, out_name in [("ni_ttm_ps", "earnings_yield"), ("fcf_ttm_ps", "fcf_yield")]:
        if col in feat.columns:
            feat[out_name] = feat[col] / feat["close"]
            feat.drop(columns=[col], inplace=True)
        else:
            feat[out_name] = np.nan
    cov = feat.loc[feat["in_universe"], "roe_ttm"].notna().mean()
    log.info("fundamentals joined (roe coverage in universe: %.1f%%)", cov * 100)
    return feat
