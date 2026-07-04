#!/usr/bin/env python3
"""
edgar_fundamentals.py — pull fundamentals from SEC EDGAR companyfacts.

Replaces the throwaway TradingView-scanner fundamentals with authoritative SEC
XBRL data. Fetches https://data.sec.gov/api/xbrl/companyfacts/CIK{n}.json per
ticker (mapped via SEC's company_tickers.json), caches the raw JSON locally, and
derives:

  eps_ttm        diluted EPS, trailing 4 quarters
  eps_yoy        YoY growth of TTM diluted EPS (%)
  revenue_ttm    revenue, trailing 4 quarters
  revenue_yoy    YoY revenue growth (%)
  net_income_ttm trailing 4-quarter net income
  equity         latest stockholders' equity (point-in-time)
  roe            net_income_ttm / equity (%)
  gross_margin   gross_profit_ttm / revenue_ttm (%)
  debt_to_equity latest total liabilities / equity
  pe             computed if a price is supplied: price / eps_ttm

Compliant with SEC fair-access rules: descriptive User-Agent, <=10 req/s.

CLI:
  python3 edgar_fundamentals.py AAPL MSFT EQT
  python3 edgar_fundamentals.py --file sp500.txt --csv report/edgar_fundamentals.csv
  python3 edgar_fundamentals.py AAPL --prices    # also fetch price for P/E via yfinance
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.request

import certifi

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "report", "edgar_cache")
TICKERMAP = os.path.join(CACHE_DIR, "company_tickers.json")
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT",
                                   "alpha research wugangc@gmail.com")}
CTX = ssl.create_default_context(cafile=certifi.where())
SEC_MIN_INTERVAL = 0.12   # ~8 req/s, under SEC's 10/s limit
_last_req = [0.0]

# candidate XBRL tags (first present wins) ----------------------------------
TAG_EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
TAG_REV = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
           "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"]
TAG_NI = ["NetIncomeLoss", "ProfitLoss"]
TAG_EQUITY = ["StockholdersEquity",
              "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
TAG_GROSS = ["GrossProfit"]
TAG_LIAB = ["Liabilities"]


def _throttle():
    dt = time.time() - _last_req[0]
    if dt < SEC_MIN_INTERVAL:
        time.sleep(SEC_MIN_INTERVAL - dt)
    _last_req[0] = time.time()


def _get_json(url, cache_path=None, max_age_days=7):
    if cache_path and os.path.exists(cache_path):
        age = (time.time() - os.path.getmtime(cache_path)) / 86400.0
        if age < max_age_days:
            with open(cache_path) as f:
                return json.load(f)
    _throttle()
    req = urllib.request.Request(url, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30, context=CTX))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
    return data


def ticker_to_cik():
    d = _get_json("https://www.sec.gov/files/company_tickers.json", TICKERMAP, max_age_days=30)
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in d.values()}


def _latest_end(fact):
    """Most recent 'end' date across all units of a fact (for tag freshness)."""
    if not fact:
        return ""
    latest = ""
    for unit in fact["units"].values():
        for r in unit:
            e = r.get("end", "")
            if e > latest:
                latest = e
    return latest


def _first_tag(gaap, tags):
    """Pick the candidate tag whose data is FRESHEST (companies switch tags over
    time, leaving stale series under an old tag; 'first present' can grab those)."""
    present = [(t, _latest_end(gaap[t])) for t in tags if t in gaap]
    if not present:
        return None
    present.sort(key=lambda x: x[1], reverse=True)
    return gaap[present[0][0]]


def _quarterly_series(fact):
    """Return sorted [(end_date, val)] of SINGLE-QUARTER values, one per quarter.
    Direct single-quarter facts are used where present; gaps are filled by
    differencing YTD-cumulative facts (Q2 = H1 - Q1, etc.) since many filers
    report some quarters only cumulatively."""
    if not fact:
        return []
    units = fact["units"]
    unit = "USD" if "USD" in units else ("USD/shares" if "USD/shares" in units
                                         else next(iter(units)))
    # bucket every period by (fiscal-year-ish anchor). We key YTD spans by their
    # start date so we can difference successive cumulative periods.
    singles = {}     # end -> single-quarter value (direct)
    ytd = {}         # (start) -> sorted list of (end, days, val) cumulative periods
    for r in fact["units"][unit]:
        s, e, v = r.get("start"), r.get("end"), r.get("val")
        if not s or not e or v is None:
            continue
        days = (_d(e) - _d(s)).days
        if 80 <= days <= 115:                    # single quarter (incl. 16-wk retail Q4)
            singles.setdefault(e, v)
        elif 115 < days <= 380:
            ytd.setdefault(s, []).append((e, days, v))
    # derive single quarters by differencing consecutive YTD periods sharing a start
    for s, periods in ytd.items():
        periods.sort(key=lambda x: x[1])   # by length
        for i in range(1, len(periods)):
            (e_prev, d_prev, v_prev) = periods[i - 1]
            (e_cur, d_cur, v_cur) = periods[i]
            if 55 <= (d_cur - d_prev) <= 115:   # one extra quarter
                singles.setdefault(e_cur, v_cur - v_prev)
    return sorted(singles.items())


def _quarterly_flow_ttm(fact):
    """TTM = sum of the 4 most recent CONSECUTIVE quarters; year-ago TTM for YoY.
    Requires the 4 (and 8) quarters to be roughly consecutive (no big gaps)."""
    q = _quarterly_series(fact)
    if len(q) < 4:
        return None, None

    def consecutive(chunk):
        # each step ~one quarter apart; allow up to 135d for retail 16-week Q4 /
        # 4-4-5 fiscal calendars, but reject a full skipped quarter (~180d+).
        for i in range(1, len(chunk)):
            if not (55 <= (_d(chunk[i][0]) - _d(chunk[i - 1][0])).days <= 135):
                return False
        return True

    last4 = q[-4:]
    ttm = sum(v for _, v in last4) if consecutive(last4) else None
    prev4 = q[-8:-4] if len(q) >= 8 else None
    ttm_prev = sum(v for _, v in prev4) if (prev4 and consecutive(prev4)) else None
    return ttm, ttm_prev


def _eps_ttm(fact):
    """TTM diluted EPS via the shared single-quarter series (handles YTD diffs),
    rounded to cents. Falls back to latest two annual (FY) EPS if quarters sparse."""
    if not fact:
        return None, None
    ttm, prev = _quarterly_flow_ttm(fact)
    if ttm is not None:
        return round(ttm, 2), (round(prev, 2) if prev is not None else None)
    # fallback: annual FY EPS
    unit = "USD/shares" if "USD/shares" in fact["units"] else next(iter(fact["units"]))
    ann = sorted({r["end"]: r["val"] for r in fact["units"][unit]
                  if r.get("fp") == "FY"}.items())
    if ann:
        return round(ann[-1][1], 2), (round(ann[-2][1], 2) if len(ann) >= 2 else None)
    return None, None


def _latest_instant(fact):
    """Latest point-in-time value (balance-sheet item)."""
    if not fact:
        return None
    unit = next(iter(fact["units"]))
    rows = [r for r in fact["units"][unit] if r.get("end")]
    if not rows:
        return None
    rows.sort(key=lambda r: r["end"])
    return rows[-1]["val"]


def _d(s):
    import datetime
    return datetime.date.fromisoformat(s)


def get_fundamentals(ticker, cik_map=None, price=None):
    cik_map = cik_map or ticker_to_cik()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return {"ticker": ticker, "error": "no CIK"}
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    cache = os.path.join(CACHE_DIR, f"CIK{cik}.json")
    try:
        cf = _get_json(url, cache)
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker, "error": str(e)[:60]}
    gaap = cf.get("facts", {}).get("us-gaap", {})

    eps_ttm, eps_prev = _eps_ttm(_first_tag(gaap, TAG_EPS))
    rev_ttm, rev_prev = _quarterly_flow_ttm(_first_tag(gaap, TAG_REV))
    ni_ttm, _ = _quarterly_flow_ttm(_first_tag(gaap, TAG_NI))
    gross_ttm, _ = _quarterly_flow_ttm(_first_tag(gaap, TAG_GROSS))
    equity = _latest_instant(_first_tag(gaap, TAG_EQUITY))
    liab = _latest_instant(_first_tag(gaap, TAG_LIAB))

    def pct(cur, prev):
        return round((cur / prev - 1) * 100, 1) if (cur is not None and prev not in (None, 0)) else None

    out = {
        "ticker": ticker.upper(),
        "eps_ttm": round(eps_ttm, 2) if eps_ttm is not None else None,
        "eps_yoy": pct(eps_ttm, eps_prev),
        "revenue_ttm": rev_ttm,
        "revenue_yoy": pct(rev_ttm, rev_prev),
        "net_income_ttm": ni_ttm,
        "equity": equity,
        "roe": round(ni_ttm / equity * 100, 1) if (ni_ttm is not None and equity) else None,
        "gross_margin": round(gross_ttm / rev_ttm * 100, 1) if (gross_ttm and rev_ttm) else None,
        "debt_to_equity": round(liab / equity, 2) if (liab is not None and equity) else None,
        "pe": round(price / eps_ttm, 1) if (price and eps_ttm and eps_ttm > 0) else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--prices", action="store_true", help="fetch price via yfinance for P/E")
    ap.add_argument("--csv")
    args = ap.parse_args()

    syms = list(args.tickers)
    if args.file:
        with open(args.file) as f:
            for line in f:
                for tok in line.replace(",", " ").split():
                    if tok.strip() and not tok.startswith("#"):
                        syms.append(tok.strip().upper())
    seen, tickers = set(), []
    for s in syms:
        if s.upper() not in seen:
            seen.add(s.upper()); tickers.append(s.upper())
    if not tickers:
        ap.error("no tickers")

    prices = {}
    if args.prices:
        import yfinance as yf
        d = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"]
        prices = {t: float(d[t].dropna().iloc[-1]) for t in (d.columns if hasattr(d, "columns") else [tickers[0]])
                  if t in getattr(d, "columns", [tickers[0]])} if len(tickers) > 1 else {tickers[0]: float(d.dropna().iloc[-1])}

    print(f"Fetching EDGAR fundamentals for {len(tickers)} tickers "
          f"(cache: {CACHE_DIR})...", file=sys.stderr)
    cik_map = ticker_to_cik()
    rows = []
    for t in tickers:
        r = get_fundamentals(t, cik_map, price=prices.get(t))
        rows.append(r)

    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[~df.get("error", pd.Series([None] * len(df))).notna()] if "error" in df else df
    show = df[["ticker", "pe", "eps_ttm", "eps_yoy", "revenue_yoy", "roe",
               "gross_margin", "debt_to_equity"]] if "pe" in df else df
    import pandas as pd
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    if "error" in df and df["error"].notna().any():
        errs = df[df["error"].notna()][["ticker", "error"]]
        print(f"\nErrors ({len(errs)}):")
        print(errs.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {len(df)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
