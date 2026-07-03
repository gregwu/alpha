#!/usr/bin/env python3
"""
alpaca_client.py — thin client all bots use to talk to trading_service.py.

Every bot imports this and calls the SAME api, so there's one place that knows
how to reach the trading service (URL + shared token). No bot talks to Alpaca
directly.

Usage in a bot:
    from alpaca_client import TradingClient
    tc = TradingClient()                 # reads TRADING_SERVICE_URL/_TOKEN from env
    if tc.enabled:
        tc.order("SSO", "buy", notional=100000)
        tc.rebalance({"AMD": 0.066, "MU": 0.066, ...})
        tc.liquidate("SSO")

If TRADING_SERVICE_URL is unset, `enabled` is False and calls are no-ops that
return None — so bots run in pure-paper (state-file) mode unless wired up.
"""
import os

import requests

DEFAULT_URL = os.environ.get("TRADING_SERVICE_URL", "")   # e.g. http://127.0.0.1:8787
DEFAULT_TOKEN = os.environ.get("TRADING_SERVICE_TOKEN", "local-dev-token")


class TradingClient:
    def __init__(self, url: str = None, token: str = None, timeout: int = 25):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.token = token or DEFAULT_TOKEN
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _post(self, path, payload):
        if not self.enabled:
            return None
        r = requests.post(self.url + path, json=payload, timeout=self.timeout,
                          headers={"X-Service-Token": self.token})
        return _safe_json(r)

    def _get(self, path):
        if not self.enabled:
            return None
        r = requests.get(self.url + path, timeout=self.timeout,
                         headers={"X-Service-Token": self.token})
        return _safe_json(r)

    # --- unified API -------------------------------------------------------
    def health(self):
        return self._get("/health")

    def account(self):
        return self._get("/account")

    def positions(self):
        return self._get("/positions")

    def order(self, symbol, side, qty=None, notional=None):
        p = {"symbol": symbol, "side": side}
        if notional is not None:
            p["notional"] = notional
        if qty is not None:
            p["qty"] = qty
        return self._post("/order", p)

    def liquidate(self, symbol):
        return self._post("/liquidate", {"symbol": symbol})

    def rebalance(self, targets: dict, cash_buffer: float = 0.02):
        return self._post("/rebalance", {"targets": targets, "cash_buffer": cash_buffer})


def _safe_json(r):
    try:
        return {"status": r.status_code, **(r.json() if r.content else {})}
    except Exception:  # noqa: BLE001
        return {"status": r.status_code, "raw": r.text}
