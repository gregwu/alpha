"""Snapshot raw data: Postgres OHLCV + metadata, yfinance benchmark."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from alpha import data

if __name__ == "__main__":
    data.snapshot_metadata()
    data.snapshot_benchmark()
    data.snapshot_prices()
    print("snapshot complete")
