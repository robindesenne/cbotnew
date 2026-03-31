#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Force V1 frozen modules
import sys
ROOT = Path(__file__).resolve().parents[1]
V1_SRC = ROOT / "versions" / "v1_frozen" / "src"
sys.path.insert(0, str(V1_SRC.parent))

from src.features import add_features  # type: ignore
from src.signals import primary_setups  # type: ignore


@dataclass
class BotState:
    cash: float
    qty: float
    entry_price: float | None
    entry_ts: str | None
    last_candle_close_time: int | None


def load_yaml(p: Path) -> dict[str, Any]:
    return yaml.safe_load(p.read_text())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_state(path: Path, initial_cash: float) -> BotState:
    if not path.exists():
        return BotState(cash=initial_cash, qty=0.0, entry_price=None, entry_ts=None, last_candle_close_time=None)
    d = json.loads(path.read_text())
    return BotState(
        cash=float(d.get("cash", initial_cash)),
        qty=float(d.get("qty", 0.0)),
        entry_price=d.get("entry_price"),
        entry_ts=d.get("entry_ts"),
        last_candle_close_time=d.get("last_candle_close_time"),
    )


def save_state(path: Path, s: BotState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "cash": s.cash,
        "qty": s.qty,
        "entry_price": s.entry_price,
        "entry_ts": s.entry_ts,
        "last_candle_close_time": s.last_candle_close_time,
    }, indent=2))


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
    for c in ["open", "high", "low", "close", "volume", "open_time", "close_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).sort_values("ts").drop_duplicates("ts")
    return df.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="live/config.paper.yaml")
    args = p.parse_args()

    cfg = load_yaml(Path(args.config))
    if cfg.get("mode") != "paper":
        print("ERROR: paper_v1.py is paper-only")
        return 1

    symbol = cfg["symbol"]
    interval = cfg["interval"]
    poll_seconds = int(cfg["execution"].get("poll_seconds", 20))
    fee_bps = float(cfg["execution"].get("fee_bps", 10))
    slip_bps = float(cfg["execution"].get("slippage_bps", 5))
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0
    max_pos_pct = float(cfg["risk"].get("max_position_pct", 1.0))

    kill_switch = ROOT / cfg["controls"]["kill_switch_file"]
    state_file = ROOT / cfg["controls"]["state_file"]
    log_file = ROOT / cfg["controls"]["log_file"]

    initial_cash = float(cfg["capital"].get("initial_quote", cfg["capital"].get("initial_usdt", 1000.0)))
    state = load_state(state_file, initial_cash)

    append_jsonl(log_file, {"event": "paper_v1_start", "symbol": symbol, "interval": interval, "initial_cash": initial_cash})

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

        cur = df.iloc[-2]  # latest closed candle
        close_time = int(cur.get("close_time", 0))
        if state.last_candle_close_time == close_time:
            time.sleep(poll_seconds)
            continue

        signal = "BUY" if int(cur.get("setup_any", 0)) == 1 else "HOLD"
        px = float(cur["close"])

        action = "NO_ACTION"
        pnl = None

        if state.qty <= 0 and signal == "BUY":
            alloc = state.cash * max_pos_pct
            qty = alloc / (px * (1 + fee + slip)) if px > 0 else 0
            notional = qty * px
            if notional >= float(cfg["risk"].get("min_notional", 10)) and qty > 0:
                cost = qty * px * (1 + fee + slip)
                state.cash -= cost
                state.qty = qty
                state.entry_price = px
                state.entry_ts = str(cur["ts"])
                action = "PAPER_BUY"

        elif state.qty > 0 and signal == "HOLD":
            # conservative phase-2 rule: flat if no setup anymore
            exit_px = px * (1 - fee - slip)
            proceeds = state.qty * exit_px
            state.cash += proceeds
            pnl = (px - float(state.entry_price or px)) * state.qty
            action = "PAPER_SELL"
            state.qty = 0.0
            state.entry_price = None
            state.entry_ts = None

        equity = state.cash + state.qty * px
        row = {
            "event": "paper_v1_tick",
            "ts": str(cur["ts"]),
            "symbol": symbol,
            "interval": interval,
            "signal": signal,
            "action": action,
            "price": px,
            "cash": state.cash,
            "qty": state.qty,
            "equity": equity,
            "pnl_est": pnl,
            "source": "v1_frozen_setup_any",
        }
        append_jsonl(log_file, row)

        state.last_candle_close_time = close_time
        save_state(state_file, state)
        time.sleep(poll_seconds)

    append_jsonl(log_file, {"event": "paper_v1_stop"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
