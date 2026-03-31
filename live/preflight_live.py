#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from binance_client import BinanceSpotClient, load_creds


def _filter_map(filters: list[dict]) -> dict[str, dict]:
    out = {}
    for f in filters:
        t = f.get("filterType")
        if t:
            out[t] = f
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="live/config.paper.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    symbol = cfg["symbol"]
    base_url = cfg["binance"]["base_url"]
    creds = load_creds(cfg["binance"]["api_key_env"], cfg["binance"]["api_secret_env"])

    c = BinanceSpotClient(base_url, creds)

    print("[1/6] ping...")
    c.ping()
    print("ok")

    print("[2/6] server time...")
    st = c.server_time()
    print("serverTime:", st)

    print("[3/6] account access...")
    acc = c.account()
    can_trade = acc.get("canTrade")
    can_withdraw = acc.get("canWithdraw")
    print("canTrade=", can_trade, "canWithdraw=", can_withdraw)

    print("[4/6] symbol info...")
    s = c.exchange_info_symbol(symbol)
    fm = _filter_map(s.get("filters", []))
    lot = fm.get("LOT_SIZE", {})
    mn = fm.get("MIN_NOTIONAL", fm.get("NOTIONAL", {}))
    print("status=", s.get("status"))
    print("LOT_SIZE:", {k: lot.get(k) for k in ["minQty", "maxQty", "stepSize"]})
    print("MIN_NOTIONAL:", {k: mn.get(k) for k in ["minNotional", "applyToMarket"]})

    print("[5/6] ticker...")
    px = c.ticker_price(symbol)
    print("price=", px)

    quote_asset = symbol[-4:] if len(symbol) >= 4 else "QUOTE"
    print(f"[6/6] sanity 20 {quote_asset} test size...")
    test_quote = 20.0
    est_qty = test_quote / px if px > 0 else 0
    print(f"test_quote=20.0 {quote_asset}", "estimated_qty=", est_qty)

    print("\nPRE-FLIGHT OK (no order placed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
