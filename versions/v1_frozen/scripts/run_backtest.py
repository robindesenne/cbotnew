#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys

import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(os.getenv("V1_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_full_dataset, run_single_backtest, save_backtest_outputs


def _algo_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    paths = [root / 'config.yaml', root / 'scripts' / 'run_backtest.py']
    paths.extend(sorted((root / 'src').glob('*.py')))
    for fp in paths:
        if fp.exists():
            h.update(fp.as_posix().encode())
            h.update(fp.read_bytes())
    return h.hexdigest()




def _prepared_cache_path(root: Path, symbol: str, interval: str, algo_fp: str) -> Path:
    cdir = root / "data" / "cache" / "prepared"
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"{symbol}_{interval}_{algo_fp[:16]}.pkl"

def _run_fingerprint(root: Path, args) -> dict:
    return {
        'algo_fp': _algo_fingerprint(root),
        'symbol': args.symbol,
        'interval': args.interval,
        'date_from': args.date_from,
        'date_to': args.date_to,
        'cash': float(args.cash),
    }



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--cash", type=float, default=10000)
    p.add_argument("--export", default="reports/backtest")
    args = p.parse_args()

    root = ROOT

    out = root / args.export
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    cur_meta = _run_fingerprint(root, args)

    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            if prev == cur_meta and (out / "summary.json").exists() and (out / "trades.csv").exists() and (out / "equity.csv").exists():
                print({"cache_hit": True, "prepared_cache_hit": True, "prepared_cache_stale": False, **cur_meta})
                return
        except Exception:
            pass

    # Prepared dataset cache (reuse overlapping periods)
    pcache = _prepared_cache_path(root, args.symbol, args.interval, cur_meta['algo_fp'])
    prepared_df = None
    prepared_source = None
    prepared_cache_hit = False
    prepared_cache_stale = False

    def _try_load_pickle(fp: Path):
        try:
            return pd.read_pickle(fp)
        except Exception:
            return None

    if pcache.exists():
        loaded = _try_load_pickle(pcache)
        if loaded is not None:
            prepared_df = loaded
            prepared_source = "prepared_cache"
            prepared_cache_hit = True

    # fallback: stale prepared cache for same symbol/interval (fast path)
    if prepared_df is None:
        pdir = root / "data" / "cache" / "prepared"
        candidates = sorted(pdir.glob(f"{args.symbol}_{args.interval}_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        for cand in candidates:
            loaded = _try_load_pickle(cand)
            if loaded is not None:
                prepared_df = loaded
                prepared_source = "prepared_cache_stale"
                prepared_cache_hit = True
                prepared_cache_stale = True
                break

    if prepared_df is None:
        prepared_df, prepared_source = prepare_full_dataset(root, args.symbol, args.interval, yaml.safe_load((root / "config.yaml").read_text()))
        try:
            prepared_df.to_pickle(pcache)
        except Exception:
            pass

    summary, trades, equity, _ = run_single_backtest(
        root=root,
        symbol=args.symbol,
        interval=args.interval,
        date_from=args.date_from,
        date_to=args.date_to,
        cash=args.cash,
        prepared_df=prepared_df,
        prepared_source=prepared_source,
    )
    save_backtest_outputs(root, args.export, summary, trades, equity)
    meta_path.write_text(json.dumps(cur_meta, indent=2))
    print({"cache_hit": False, "prepared_cache_hit": prepared_cache_hit, "prepared_cache_stale": prepared_cache_stale, **summary})


if __name__ == "__main__":
    main()
