#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "versions" / "v1_frozen").resolve()))

from src.features import add_features  # type: ignore
from src.signals import primary_setups  # type: ignore

from binance_client import BinanceSpotClient, load_creds


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_market_df(project_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    base = project_dir / "data" / "market" / "binance_spot" / symbol
    files = sorted(base.glob(f"{symbol}_{interval}_*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "ts" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "close_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).sort_values("ts").drop_duplicates("ts")
    return df.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="live/config.live.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("mode") != "live":
        print("ERROR: mode must be live")
        return 1

    symbol = cfg["symbol"]
    interval = cfg["interval"]
    poll_seconds = int(cfg["execution"].get("poll_seconds", 20))
    min_notional = float(cfg["risk"].get("min_notional", 10))
    test_quote_usdt = float(cfg["execution"].get("test_quote", cfg["execution"].get("test_quote_usdt", 20.0)))
    hard_max_quote = float(cfg["execution"].get("hard_max_quote_per_order", test_quote_usdt))
    max_use_pct = float(cfg["execution"].get("max_use_of_free_quote_pct", 1.0))
    enable_live_orders = bool(cfg["execution"].get("enable_live_orders", False))

    kill_switch = ROOT / cfg["controls"]["kill_switch_file"]
    arm_file = ROOT / cfg["controls"].get("arm_file", "live/ARMED")
    log_file = ROOT / cfg["controls"]["log_file"]

    creds = load_creds(cfg["binance"]["api_key_env"], cfg["binance"]["api_secret_env"])
    bc = BinanceSpotClient(cfg["binance"]["base_url"], creds)

    # simple local state
    state_file = ROOT / cfg["controls"]["state_file"]
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        state = {"in_position": False, "qty": 0.0, "last_close_time": None}

    append_jsonl(log_file, {"event": "live_v1_start", "symbol": symbol, "interval": interval, "enable_live_orders": enable_live_orders})

    while True:
        if kill_switch.exists():
            append_jsonl(log_file, {"event": "kill_switch_detected"})
            break

        df = load_market_df(ROOT, symbol, interval)
        if len(df) < 400:
            append_jsonl(log_file, {"event": "warn", "msg": "not_enough_data", "rows": int(len(df))})
            time.sleep(poll_seconds)
            continue

        df = add_features(df)
        setups = primary_setups(df)
        df = pd.concat([df, setups], axis=1)

        cur = df.iloc[-2]
        close_time = int(cur.get("close_time", 0))
        if state.get("last_close_time") == close_time:
            time.sleep(poll_seconds)
            continue

        signal_buy = int(cur.get("setup_any", 0)) == 1
        px = float(cur["close"])

        action = "NO_ACTION"
        order_resp = None
        free_quote = None
        quote_to_use = None

        # Safety gates: live enabled + armed file required
        armed = arm_file.exists()
        can_send = enable_live_orders and armed

        if (not state.get("in_position")) and signal_buy:
            quote_asset = symbol[-4:] if len(symbol) >= 4 else "USDT"
            free_quote = None
            try:
                acc = bc.account()
                bal = [b for b in acc.get("balances", []) if b.get("asset") == quote_asset]
                free_quote = float(bal[0].get("free", 0.0)) if bal else 0.0
            except Exception:
                free_quote = None

            if test_quote_usdt > hard_max_quote:
                action = "BUY_BLOCKED_HARD_CAP"
            else:
                quote_to_use = test_quote_usdt
                if free_quote is not None:
                    quote_to_use = min(quote_to_use, free_quote * max_use_pct)

                if quote_to_use >= min_notional:
                    if can_send:
                        order_resp = bc.market_buy_quote(symbol, quote_to_use)
                        executed_qty = float(order_resp.get("executedQty", 0.0))
                        state["in_position"] = executed_qty > 0
                        state["qty"] = executed_qty
                        action = "LIVE_BUY"
                    else:
                        action = "BUY_BLOCKED_GUARDRAIL"
                else:
                    action = "BUY_BLOCKED_MIN_NOTIONAL"

        elif state.get("in_position") and (not signal_buy):
            qty = float(state.get("qty", 0.0))
            if qty > 0:
                if can_send:
                    order_resp = bc.market_sell_qty(symbol, qty)
                    state["in_position"] = False
                    state["qty"] = 0.0
                    action = "LIVE_SELL"
                else:
                    action = "SELL_BLOCKED_GUARDRAIL"

        state["last_close_time"] = close_time
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))

        append_jsonl(log_file, {
            "event": "live_v1_tick",
            "ts": str(cur["ts"]),
            "symbol": symbol,
            "interval": interval,
            "signal_buy": signal_buy,
            "price": px,
            "action": action,
            "guardrail_armed_file": str(arm_file),
            "guardrail_armed": armed,
            "live_orders_enabled": enable_live_orders,
            "in_position": state.get("in_position"),
            "qty": state.get("qty"),
            "order": order_resp,
            "source": "v1_frozen_setup_any",
            "test_quote": test_quote_usdt,
            "hard_max_quote_per_order": hard_max_quote,
            "max_use_of_free_quote_pct": max_use_pct,
            "free_quote_balance": free_quote if 'free_quote' in locals() else None,
            "quote_to_use": quote_to_use if 'quote_to_use' in locals() else None,
        })

        time.sleep(poll_seconds)

    append_jsonl(log_file, {"event": "live_v1_stop"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
