"""Build point-in-time fundamental features from SEC EDGAR companyfacts.

Prereq: data/companyfacts.zip (bulk download from
https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip).
Refresh the zip + re-run monthly; quarterly filings drive the features.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd

from alpha.config import DATA_DIR
from alpha.features.fundamentals import (FUND_RAW_PATH, build_fundamental_features,
                                         parse_companyfacts)

if __name__ == "__main__":
    prices = pd.read_parquet(DATA_DIR / "prices.parquet", columns=["ticker"])
    tickers_base = {t.replace(".US", "") for t in prices["ticker"].unique()}

    if "--reuse-raw" in sys.argv and FUND_RAW_PATH.exists():
        raw = pd.read_parquet(FUND_RAW_PATH)
    else:
        raw = parse_companyfacts(DATA_DIR / "companyfacts.zip", tickers_base)
    build_fundamental_features(raw)
