#!/usr/bin/env python3
"""
Télécharge et stocke l'historique Binance Spot klines pour backtests.

Exemple:
python3 scripts/download_binance_history.py \
  --symbols BTCUSDC ETHUSDC SOLUSDC \
  --intervals 15m 1h 4h \
  --from 2022-01-01 --to 2026-03-21
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"


def iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    out = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        cursor = int(data[-1][0]) + 1
        if len(data) < 1000:
            break
        time.sleep(0.15)
    return out


def to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    for c in ["open_time", "close_time", "number_of_trades"]:
        df[c] = df[c].astype("int64")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--intervals", nargs="+", default=["15m", "1h"])
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--out", default="data/market/binance_spot")
    args = p.parse_args()

    start_ms = iso_to_ms(args.date_from)
    end_ms = iso_to_ms(args.date_to)

    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for symbol in args.symbols:
        for interval in args.intervals:
            print(f"Downloading {symbol} {interval}...")
            raw = fetch_klines(symbol, interval, start_ms, end_ms)
            if not raw:
                print(f"  -> no data")
                summary.append((symbol, interval, 0, "no-data"))
                continue

            df = to_df(raw)
            # drop duplicate candle opens if any
            df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

            sym_dir = out_root / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            csv_path = sym_dir / f"{symbol}_{interval}_{args.date_from}_{args.date_to}.csv"
            parquet_path = sym_dir / f"{symbol}_{interval}_{args.date_from}_{args.date_to}.parquet"

            df.to_csv(csv_path, index=False)
            try:
                df.to_parquet(parquet_path, index=False)
                pq_state = "ok"
            except Exception:
                pq_state = "skip"

            summary.append((symbol, interval, len(df), pq_state))
            print(f"  -> {len(df)} rows saved")

    print("\nSummary:")
    for s in summary:
        print({"symbol": s[0], "interval": s[1], "rows": s[2], "parquet": s[3]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
