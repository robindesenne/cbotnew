#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests


@dataclass
class BinanceCreds:
    api_key: str
    api_secret: str


def load_creds(key_env: str, secret_env: str) -> BinanceCreds:
    k = os.getenv(key_env or "")
    s = os.getenv(secret_env or "")
    if not k or not s:
        raise RuntimeError(f"Missing API creds in env: {key_env}/{secret_env}")
    return BinanceCreds(api_key=k, api_secret=s)


def _sign(secret: str, qs: str) -> str:
    return hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()


class BinanceSpotClient:
    def __init__(self, base_url: str, creds: BinanceCreds):
        self.base_url = base_url.rstrip("/")
        self.creds = creds

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.creds.api_key}

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = requests.get(f"{self.base_url}{path}", params=params or {}, timeout=20)
        r.raise_for_status()
        return r.json()

    def signed_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        p = dict(params or {})
        p["timestamp"] = int(time.time() * 1000)
        qs = urlencode(p, doseq=True)
        p["signature"] = _sign(self.creds.api_secret, qs)
        r = requests.get(f"{self.base_url}{path}", params=p, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def signed_post(self, path: str, params: dict[str, Any] | None = None) -> Any:
        p = dict(params or {})
        p["timestamp"] = int(time.time() * 1000)
        qs = urlencode(p, doseq=True)
        p["signature"] = _sign(self.creds.api_secret, qs)
        r = requests.post(f"{self.base_url}{path}", params=p, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def ping(self) -> bool:
        self.public_get("/api/v3/ping")
        return True

    def server_time(self) -> int:
        return int(self.public_get("/api/v3/time")["serverTime"])

    def account(self) -> dict[str, Any]:
        return self.signed_get("/api/v3/account")

    def exchange_info_symbol(self, symbol: str) -> dict[str, Any]:
        data = self.public_get("/api/v3/exchangeInfo", {"symbol": symbol})
        syms = data.get("symbols", [])
        if not syms:
            raise RuntimeError(f"symbol not found: {symbol}")
        return syms[0]

    def ticker_price(self, symbol: str) -> float:
        data = self.public_get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def market_buy_quote(self, symbol: str, quote_qty: float) -> dict[str, Any]:
        return self.signed_post("/api/v3/order", {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": f"{quote_qty:.8f}",
            "newOrderRespType": "FULL",
        })

    def market_sell_qty(self, symbol: str, qty: float) -> dict[str, Any]:
        return self.signed_post("/api/v3/order", {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": f"{qty:.8f}",
            "newOrderRespType": "FULL",
        })
