#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


def load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def now_ts() -> int:
    return int(time.time())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_last_closed_candle(base_url: str, symbol: str, interval: str) -> Candle:
    r = requests.get(
        f"{base_url}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": 3},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    # penultimate candle = dernière bougie close sûre
    c = data[-2]
    return Candle(
        open_time=int(c[0]), open=float(c[1]), high=float(c[2]), low=float(c[3]), close=float(c[4]),
        volume=float(c[5]), close_time=int(c[6])
    )


def simple_signal(prev_close: float | None, cur_close: float) -> str:
    if prev_close is None:
        return "HOLD"
    if cur_close > prev_close:
        return "BUY"
    if cur_close < prev_close:
        return "SELL"
    return "HOLD"


def run(config_path: Path) -> None:
    cfg = load_cfg(config_path)
    symbol = cfg["symbol"]
    interval = cfg["interval"]
    poll_seconds = int(cfg["execution"]["poll_seconds"])
    kill_switch = Path(cfg["controls"]["kill_switch_file"])
    log_file = Path(cfg["controls"]["log_file"])
    base_url = cfg["binance"]["base_url"]

    mode = cfg.get("mode", "paper")
    enable_live_orders = bool(cfg["execution"].get("enable_live_orders", False))

    prev_close = None
    last_candle_close_time = None

    append_jsonl(log_file, {"ts": now_ts(), "event": "bot_start", "mode": mode, "symbol": symbol, "interval": interval})

    while True:
        if kill_switch.exists():
            append_jsonl(log_file, {"ts": now_ts(), "event": "kill_switch_detected", "path": str(kill_switch)})
            break

        try:
            c = fetch_last_closed_candle(base_url, symbol, interval)
            if last_candle_close_time == c.close_time:
                time.sleep(poll_seconds)
                continue

            sig = simple_signal(prev_close, c.close)
            row = {
                "ts": now_ts(),
                "event": "decision",
                "mode": mode,
                "symbol": symbol,
                "interval": interval,
                "candle_close_time": c.close_time,
                "price_close": c.close,
                "signal": sig,
                "live_orders_enabled": enable_live_orders,
                "action": "NO_ORDER" if (mode != "live" or not enable_live_orders) else "ORDER_PLACEHOLDER",
            }
            append_jsonl(log_file, row)
            prev_close = c.close
            last_candle_close_time = c.close_time

        except Exception as e:
            append_jsonl(log_file, {"ts": now_ts(), "event": "error", "error": str(e)})

        time.sleep(poll_seconds)

    append_jsonl(log_file, {"ts": now_ts(), "event": "bot_stop"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="live/config.paper.yaml")
    args = p.parse_args()
    run(Path(args.config))
