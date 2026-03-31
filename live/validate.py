#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


REQUIRED_TOP = ["mode", "symbol", "interval", "capital", "risk", "execution", "controls", "binance"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="live/config.paper.yaml")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        return 1

    cfg = yaml.safe_load(cfg_path.read_text())

    for k in REQUIRED_TOP:
        if k not in cfg:
            print(f"ERROR: missing key: {k}")
            return 1

    mode = cfg["mode"]
    if mode not in {"paper", "live"}:
        print("ERROR: mode must be paper|live")
        return 1

    if mode == "live":
        k = cfg["binance"].get("api_key_env")
        s = cfg["binance"].get("api_secret_env")
        if not os.getenv(k or "") or not os.getenv(s or ""):
            print("ERROR: live mode requires API env vars")
            return 1

    print("OK: configuration is valid")
    print(f"- mode={mode}")
    print(f"- symbol={cfg['symbol']} interval={cfg['interval']}")
    print(f"- live_orders_enabled={cfg['execution'].get('enable_live_orders')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
