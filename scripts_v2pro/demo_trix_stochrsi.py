#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_v2.data_loader import LoadSpec, load_ohlcv
from src_v2pro.indicator_trix_stochrsi import add_trix_stochrsi_combo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--out", default="reports/trix_stochrsi_demo.csv")
    args = p.parse_args()

    df, source = load_ohlcv(ROOT, LoadSpec(symbol=args.symbol, interval=args.interval, date_from=args.date_from, date_to=args.date_to, market_type="spot"))
    out = add_trix_stochrsi_combo(df)

    cols = [
        "ts", "close", "trix", "trix_signal", "trix_hist",
        "stoch_rsi_k", "stoch_rsi_d", "combo_long", "combo_exit",
    ]
    out = out[[c for c in cols if c in out.columns]].copy()
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / args.out, index=False)

    print({
        "symbol": args.symbol,
        "interval": args.interval,
        "source": source,
        "rows": int(len(out)),
        "combo_long_count": int(out["combo_long"].sum()) if "combo_long" in out else 0,
        "combo_exit_count": int(out["combo_exit"].sum()) if "combo_exit" in out else 0,
        "out": str(ROOT / args.out),
    })


if __name__ == "__main__":
    main()
