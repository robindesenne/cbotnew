#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.backtest_engine import run_single_backtest, save_backtest_outputs
from src.backtest.config import GLOBAL_CONFIG
from src.backtest.solusdt_pipeline import SolusdtPipelineSpec, build_solusdt_dataset
from src.strategies import StrategyContext, registry


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_one(
    strategy_name: str,
    interval: str,
    date_from: str,
    date_to: str,
    cash: float,
    out_root: Path,
    base_ohlcv: pd.DataFrame,
) -> Dict:
    spec = registry.get(strategy_name)
    strategy = spec.build()

    ctx = StrategyContext(
        symbol=GLOBAL_CONFIG["solusdt"]["symbol"],
        interval=interval,
        horizon_bars=GLOBAL_CONFIG["label_horizon_bars"],
        config={},
    )

    # strategy-level prepared dataset (includes features/labels/regime via BaseStrategy)
    prepared = strategy.generate_signals(base_ohlcv, ctx)
    if "signal" in prepared.columns:
        prepared["setup_any"] = (prepared["signal"] != 0).astype(int)
    elif "trade_flag" in prepared.columns:
        prepared["setup_any"] = prepared["trade_flag"].astype(int)

    summary, trades, equity, _ = run_single_backtest(
        root=ROOT,
        symbol=GLOBAL_CONFIG["solusdt"]["symbol"],
        interval=interval,
        date_from=date_from,
        date_to=date_to,
        cash=cash,
        prepared_df=prepared,
        prepared_source=f"strategy:{strategy_name}",
    )

    summary["strategy_name"] = strategy_name
    summary["strategy_family"] = spec.family
    summary["strategy_version"] = spec.version

    out_dir = out_root / strategy_name
    save_backtest_outputs(ROOT, str(out_dir.relative_to(ROOT)), summary, trades, equity)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Ultra Benchmark SOLUSDT runner (40 strategies)")
    p.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--interval", default=GLOBAL_CONFIG["solusdt"]["default_timeframe"])
    p.add_argument("--date-from", default=GLOBAL_CONFIG["solusdt"]["history_start"])
    p.add_argument("--date-to", default="2026-12-31")
    p.add_argument("--limit", type=int, default=0, help="Limit number of strategies (0 = all)")
    p.add_argument("--offset", type=int, default=0, help="Start index in sorted strategy list")
    p.add_argument("--resume", action="store_true", help="Resume from existing summary_all.csv (skip completed strategies)")
    args = p.parse_args()

    out_root = ROOT / "reports" / "ultra_benchmark_sol" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    pipe_spec = SolusdtPipelineSpec(interval=args.interval, date_from=args.date_from, date_to=args.date_to)
    base_ds, source = build_solusdt_dataset(ROOT, pipe_spec)
    base_ohlcv = base_ds[["ts", "open", "high", "low", "close", "volume"]].copy()

    strategies = sorted(set(registry.list_names()))
    if args.offset > 0:
        strategies = strategies[args.offset:]
    if args.limit > 0:
        strategies = strategies[: args.limit]

    summary_all_path = out_root / "summary_all.csv"
    existing_results: List[Dict] = []
    done = set()
    if args.resume and summary_all_path.exists():
        try:
            exdf = pd.read_csv(summary_all_path)
            existing_results = exdf.to_dict(orient="records")
            done = set(exdf.get("strategy_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
        except Exception:
            existing_results = []
            done = set()

    strategies = [s for s in strategies if s not in done]

    run_meta = {
        "run_id": args.run_id,
        "started_at": now_iso(),
        "symbol": GLOBAL_CONFIG["solusdt"]["symbol"],
        "interval": args.interval,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "strategy_count": len(strategies),
        "max_workers": args.max_workers,
        "source": source,
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    print(f"[ultra-benchmark] run_id={args.run_id} strategies={len(strategies)} workers={args.max_workers}")

    results: List[Dict] = list(existing_results)

    def checkpoint_write():
        cdf = pd.DataFrame(results)
        cdf.to_csv(summary_all_path, index=False)

    if args.max_workers <= 1:
        for i, name in enumerate(strategies, start=1):
            print(f"[{i}/{len(strategies)}] {name} ...")
            try:
                res = _run_one(name, args.interval, args.date_from, args.date_to, args.cash, out_root, base_ohlcv)
                results.append(res)
                checkpoint_write()
                print(f"    done trades={res.get('trades',0)} sharpe={res.get('sharpe',0):.3f} pf={res.get('profit_factor',0):.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"strategy_name": name, "error": str(e)})
                checkpoint_write()
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(_run_one, name, args.interval, args.date_from, args.date_to, args.cash, out_root, base_ohlcv): name
                for name in strategies
            }
            done = 0
            for fut in as_completed(futs):
                name = futs[fut]
                done += 1
                try:
                    res = fut.result()
                    results.append(res)
                    checkpoint_write()
                    print(f"[{done}/{len(strategies)}] {name} done sharpe={res.get('sharpe',0):.3f} pf={res.get('profit_factor',0):.3f}")
                except Exception as e:
                    print(f"[{done}/{len(strategies)}] {name} ERROR: {e}")
                    results.append({"strategy_name": name, "error": str(e)})
                    checkpoint_write()

    summary_df = pd.DataFrame(results)
    if "error" in summary_df.columns:
        ok = summary_df[summary_df["error"].isna()] if summary_df["error"].dtype != object else summary_df[~summary_df["error"].astype(str).str.len().gt(0)]
    else:
        ok = summary_df

    if not ok.empty and "sharpe" in ok.columns:
        rank = ok.sort_values(["sharpe", "profit_factor", "calmar"], ascending=False)
    else:
        rank = summary_df

    summary_df.to_csv(out_root / "summary_all.csv", index=False)
    rank.to_csv(out_root / "summary_ranked.csv", index=False)

    final = {
        **run_meta,
        "finished_at": now_iso(),
        "output_dir": str(out_root),
        "n_results": int(len(summary_df)),
    }
    (out_root / "summary.json").write_text(json.dumps(final, indent=2))

    print(f"[ultra-benchmark] finished. outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
