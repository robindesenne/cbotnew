#!/usr/bin/env python3
"""
Construit un fichier compact exploitable par la webapp à partir d'un run ultra benchmark.

Outputs:
- reports/ultra_benchmark_sol/<run_id>/micro_results_compact.csv
- reports/ultra_benchmark_sol/<run_id>/buyhold_ref.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import LoadSpec, load_ohlcv


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _buyhold_capital(symbol: str, interval: str, date_from: str, date_to: str, cash0: float) -> tuple[float, float]:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    fee = float(cfg["exchange"]["taker_fee"])
    slip = float(cfg["exchange"]["slippage_bps"]) / 10000.0

    spec = LoadSpec(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to, market_type="spot")
    df, _ = load_ohlcv(ROOT, spec)
    if len(df) < 2:
        return cash0, 0.0

    entry = float(df.iloc[1]["open"]) * (1 + slip)
    qty = cash0 / (entry * (1 + fee))
    exit_px = float(df.iloc[-1]["close"]) * (1 - slip)
    bh_cap = qty * exit_px * (1 - fee)
    bh_ret = (bh_cap / cash0 - 1.0) * 100.0
    return bh_cap, bh_ret


def build_compact(run_id: str, cash: float | None, symbol: str | None, interval: str | None, date_from: str | None, date_to: str | None) -> Path:
    run_dir = ROOT / "reports" / "ultra_benchmark_sol" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run introuvable: {run_dir}")

    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    symbol = symbol or meta.get("symbol") or "SOLUSDT"
    interval = interval or meta.get("interval") or "1h"
    date_from = date_from or meta.get("date_from")
    date_to = date_to or meta.get("date_to")
    cash0 = float(cash if cash is not None else meta.get("cash", 10000.0))

    if not date_from or not date_to:
        raise ValueError("date_from/date_to absents: fournis-les en argument ou via run_meta.json")

    rows = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        sj = child / "summary.json"
        eq = child / "equity.csv"
        tr = child / "trades.csv"
        if not sj.exists() or not eq.exists():
            continue

        try:
            summary = json.loads(sj.read_text())
        except Exception:
            continue

        cap = float("nan")
        try:
            edf = pd.read_csv(eq)
            cap = float(pd.to_numeric(edf["equity"], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass

        fees = 0.0
        if tr.exists():
            try:
                tdf = pd.read_csv(tr)
                fees = float(pd.to_numeric(tdf.get("fees", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            except Exception:
                pass

        ret = (cap / cash0 - 1.0) * 100.0 if np.isfinite(cap) else float("nan")
        rows.append(
            {
                "strategy_name": str(summary.get("strategy_name") or child.name),
                "run_id": run_id,
                "trades": _safe_float(summary.get("trades")),
                "sharpe": _safe_float(summary.get("sharpe")),
                "profit_factor": _safe_float(summary.get("profit_factor")),
                "max_drawdown": _safe_float(summary.get("max_drawdown")),
                "capital_final": cap,
                "return_pct": ret,
                "total_return_pct": ret,
                "fees_total_usd": fees,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"Aucun résultat exploitable trouvé dans {run_dir}")

    bh_cap, bh_ret = _buyhold_capital(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to, cash0=cash0)
    out["delta_vs_bh_pct_pts"] = out["return_pct"] - bh_ret
    out["delta_vs_buyhold_pct"] = out["total_return_pct"] - bh_ret
    out["delta_vs_buyhold_usd"] = out["capital_final"] - bh_cap
    out["better_than_buyhold"] = np.where(out["capital_final"] > bh_cap, "yes", "no")

    out = out.sort_values("capital_final", ascending=False).reset_index(drop=True)
    out_path = run_dir / "micro_results_compact.csv"
    out.to_csv(out_path, index=False)

    (run_dir / "buyhold_ref.txt").write_text(
        f"capital_final={bh_cap:.6f}\n"
        f"return_pct={bh_ret:.6f}\n"
    )

    print(f"[ok] compact publié: {out_path}")
    print(f"[ok] buy&hold: {bh_cap:.2f} ({bh_ret:+.2f}%)")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Publie micro_results_compact.csv pour un run benchmark")
    p.add_argument("--run-id", required=True)
    p.add_argument("--cash", type=float, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--interval", default=None)
    p.add_argument("--date-from", default=None)
    p.add_argument("--date-to", default=None)
    args = p.parse_args()

    build_compact(
        run_id=args.run_id,
        cash=args.cash,
        symbol=args.symbol,
        interval=args.interval,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
