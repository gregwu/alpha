#!/usr/bin/env python3
"""
trading_service.py — a single Alpaca trading webservice that all bots call.

Exposes one unified HTTP API so rsi_pivot_bot / lev_trend_bot / momentum_bot
(and anything else) place orders through the same endpoint instead of each
talking to Alpaca directly.

SAFETY:
  - Defaults to PAPER trading (https://paper-api.alpaca.markets).
  - Only uses the LIVE endpoint if ALPACA_LIVE=1 is explicitly set in the env.
  - Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from .env (never logged).
  - The service itself requires a shared SERVICE_TOKEN header so random callers
    on localhost can't submit orders.

Endpoints:
  GET  /health                     -> {ok, mode, market_open}
  GET  /account                    -> buying power, equity, cash
  GET  /positions                  -> current positions
  POST /order      {symbol, side, qty|notional, type, time_in_force}
  POST /rebalance  {targets:{SYM:weight,...}, cash_buffer}   (momentum bot)
  POST /liquidate  {symbol}                                   (close one)

Run:
  python3 trading_service.py                 # paper, port 8787
  PORT=9000 python3 trading_service.py
  ALPACA_LIVE=1 python3 trading_service.py   # LIVE (real money) — explicit only
"""
import os
import sys

import requests
from flask import Flask, jsonify, request

# ---- load .env (simple parser, no extra dep) -------------------------------
def _load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
LIVE = os.environ.get("ALPACA_LIVE", "") == "1"
BASE = "https://api.alpaca.markets" if LIVE else "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"
MODE = "LIVE" if LIVE else "PAPER"
# shared token bots must send; defaults to a value the bots also default to
SERVICE_TOKEN = os.environ.get("TRADING_SERVICE_TOKEN", "local-dev-token")

HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

app = Flask(__name__)


def _auth_ok(req):
    return req.headers.get("X-Service-Token", "") == SERVICE_TOKEN


def _alp(method, path, base=BASE, **kw):
    r = requests.request(method, base + path, headers=HEADERS, timeout=20, **kw)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {"raw": r.text}
    return r.status_code, body


@app.before_request
def _guard():
    if request.endpoint == "health":
        return None
    if not API_KEY or not SECRET_KEY:
        return jsonify(error="Alpaca keys not configured in .env"), 500
    if not _auth_ok(request):
        return jsonify(error="bad or missing X-Service-Token"), 401
    return None


@app.get("/health")
def health():
    ok = bool(API_KEY and SECRET_KEY)
    market_open = None
    if ok:
        code, clock = _alp("GET", "/v2/clock")
        market_open = clock.get("is_open") if code == 200 else None
    return jsonify(ok=ok, mode=MODE, base=BASE, market_open=market_open)


@app.get("/account")
def account():
    code, body = _alp("GET", "/v2/account")
    if code != 200:
        return jsonify(error="alpaca", detail=body), code
    return jsonify(mode=MODE, equity=body.get("equity"), cash=body.get("cash"),
                   buying_power=body.get("buying_power"), status=body.get("status"))


@app.get("/positions")
def positions():
    code, body = _alp("GET", "/v2/positions")
    if code != 200:
        return jsonify(error="alpaca", detail=body), code
    out = [{"symbol": p["symbol"], "qty": p["qty"], "avg_entry": p["avg_entry_price"],
            "market_value": p["market_value"], "unrealized_plpc": p["unrealized_plpc"]}
           for p in body]
    return jsonify(positions=out, count=len(out))


@app.post("/order")
def order():
    d = request.get_json(force=True) or {}
    sym = d.get("symbol")
    side = d.get("side")
    if not sym or side not in ("buy", "sell"):
        return jsonify(error="need symbol and side=buy|sell"), 400
    payload = {"symbol": sym, "side": side,
               "type": d.get("type", "market"),
               "time_in_force": d.get("time_in_force", "day")}
    if d.get("notional") is not None:
        payload["notional"] = str(d["notional"])
    elif d.get("qty") is not None:
        payload["qty"] = str(d["qty"])
    else:
        return jsonify(error="need qty or notional"), 400
    code, body = _alp("POST", "/v2/orders", json=payload)
    if code >= 300:
        return jsonify(error="alpaca", detail=body), code
    return jsonify(ok=True, id=body.get("id"), symbol=sym, side=side,
                   qty=body.get("qty"), notional=body.get("notional"),
                   status=body.get("status"))


@app.post("/liquidate")
def liquidate():
    d = request.get_json(force=True) or {}
    sym = d.get("symbol")
    if not sym:
        return jsonify(error="need symbol"), 400
    code, body = _alp("DELETE", f"/v2/positions/{sym}")
    if code >= 300:
        return jsonify(error="alpaca", detail=body), code
    return jsonify(ok=True, closed=sym, detail=body)


@app.post("/rebalance")
def rebalance():
    """
    Set the account to a target weighting. targets={SYM: weight,...} (weights
    need not sum to 1; the remainder stays cash). Sells positions not in targets,
    then market-buys each target to (equity*weight) using notional orders.
    Simple version: liquidate-then-buy (fine for weekly rebalances).
    """
    d = request.get_json(force=True) or {}
    targets = d.get("targets") or {}
    if not targets:
        return jsonify(error="need targets"), 400
    cash_buffer = float(d.get("cash_buffer", 0.02))  # keep 2% cash for slippage

    code, acct = _alp("GET", "/v2/account")
    if code != 200:
        return jsonify(error="account", detail=acct), code
    equity = float(acct["equity"])

    code, held = _alp("GET", "/v2/positions")
    held_syms = {p["symbol"] for p in held} if code == 200 else set()

    results = {"sold": [], "bought": [], "errors": []}
    # 1) sell anything not in the new targets
    for s in held_syms - set(targets):
        c, b = _alp("DELETE", f"/v2/positions/{s}")
        (results["sold"] if c < 300 else results["errors"]).append(s)

    # 2) buy/adjust each target by notional = equity * weight * (1-buffer)
    for s, w in targets.items():
        notional = round(equity * float(w) * (1 - cash_buffer), 2)
        if notional <= 1:
            continue
        # naive: for a fresh weekly book we just buy; existing holders are left
        # (a production version would diff to target). Skip if already held.
        if s in held_syms:
            continue
        c, b = _alp("POST", "/v2/orders",
                    json={"symbol": s, "notional": str(notional),
                          "side": "buy", "type": "market", "time_in_force": "day"})
        (results["bought"] if c < 300 else results["errors"]).append(
            s if c < 300 else {"sym": s, "detail": b})
    return jsonify(ok=True, mode=MODE, equity=equity, **results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8787))
    print(f"[trading_service] mode={MODE} base={BASE} port={port} "
          f"keys={'set' if API_KEY else 'MISSING'}", file=sys.stderr)
    if LIVE:
        print("[trading_service] *** LIVE TRADING — REAL MONEY ***", file=sys.stderr)
    app.run(host="127.0.0.1", port=port, debug=False)
