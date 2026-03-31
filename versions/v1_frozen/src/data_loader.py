from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import pandas as pd
import requests


@dataclass
class LoadSpec:
    symbol: str
    interval: str
    date_from: str
    date_to: str
    data_root: str = "data/market/binance_spot"
    prefer_local: bool = True
    market_type: str = "spot"


def _parse_utc(s: str, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(str(s))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if end_of_day and len(str(s)) <= 10:
        ts += pd.Timedelta(days=1)
    return ts


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def load_local_ohlcv(root: Path, spec: LoadSpec) -> Optional[pd.DataFrame]:
    base = (root / spec.data_root).resolve() / spec.symbol.upper()
    if not base.exists():
        return None
    files = sorted(base.glob(f"{spec.symbol.upper()}_{spec.interval}_*.csv"))
    if not files:
        return None

    # choose widest coverage file (earliest start + latest end), not lexicographic last
    chosen = None
    best_span = None
    pat = re.compile(rf"^{spec.symbol.upper()}_{spec.interval}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        try:
            d0 = pd.Timestamp(m.group(1), tz='UTC')
            d1 = pd.Timestamp(m.group(2), tz='UTC')
            span = d1 - d0
            if best_span is None or span > best_span:
                best_span = span
                chosen = f
        except Exception:
            continue

    if chosen is None:
        chosen = files[-1]

    df = pd.read_csv(chosen)
    if "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts")

    d0 = _parse_utc(spec.date_from)
    d1 = _parse_utc(spec.date_to, end_of_day=True)
    df = df[(df["ts"] >= d0) & (df["ts"] < d1)].copy()
    if df.empty:
        return None

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_binance_ohlcv(spec: LoadSpec) -> pd.DataFrame:
    endpoint = (
        "https://api.binance.com/api/v3/klines"
        if spec.market_type == "spot"
        else "https://fapi.binance.com/fapi/v1/klines"
    )

    start_ms = _to_ms(_parse_utc(spec.date_from))
    end_ms = _to_ms(_parse_utc(spec.date_to, end_of_day=True))

    out = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": spec.symbol.upper(),
            "interval": spec.interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        cur = int(data[-1][0]) + 1
        if len(data) < 1000:
            break

    if not out:
        raise RuntimeError("No OHLCV data")

    df = pd.DataFrame(
        out,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)


# V1_FROZEN_STRICT_LOCAL
def load_ohlcv(root: Path, spec: LoadSpec) -> tuple[pd.DataFrame, str]:
    if spec.prefer_local:
        local = load_local_ohlcv(root, spec)
        if local is not None:
            return local, "local_csv"
        raise RuntimeError(f"V1 frozen strict-local: missing local OHLCV for {spec.symbol} {spec.interval}")
    return fetch_binance_ohlcv(spec), "binance_api"
