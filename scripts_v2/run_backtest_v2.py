#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_v2.backtest import run_backtest_v2, save_outputs


def _algo_fp(root: Path) -> str:
    h = hashlib.sha256()
    paths = [root / "config_v2.yaml", root / "scripts_v2" / "run_backtest_v2.py"]
    paths.extend(sorted((root / "src_v2").glob("*.py")))
    for p in paths:
        if p.exists():
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--cash", type=float, default=10000)
    p.add_argument("--export", default="reports/backtest")
    args = p.parse_args()

    out = ROOT / args.export
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "algo_fp": _algo_fp(ROOT),
        "symbol": args.symbol,
        "interval": args.interval,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "cash": float(args.cash),
    }
    meta_path = out / "run_meta.json"
    if meta_path.exists() and (out / "summary.json").exists() and (out / "trades.csv").exists() and (out / "equity.csv").exists():
        try:
            prev = json.loads(meta_path.read_text())
            if prev == meta:
                print({"cache_hit": True, **meta})
                return
        except Exception:
            pass

    summary, trades, equity = run_backtest_v2(ROOT, args.symbol, args.interval, args.date_from, args.date_to, args.cash)
    save_outputs(ROOT, args.export, summary, trades, equity)
    meta_path.write_text(json.dumps(meta, indent=2))
    print({"cache_hit": False, **summary})


if __name__ == "__main__":
    main()
