#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from src.strategies import registry


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit leakage + realism for strategy benchmark output")
    ap.add_argument("--run-id", default="ultra-sol-v1-full-r7")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--maxdd-threshold", type=float, default=0.60)
    ap.add_argument("--sharpe-anomaly", type=float, default=6.0)
    ap.add_argument("--calmar-anomaly", type=float, default=20.0)
    ap.add_argument("--min-trades", type=int, default=50)
    args = ap.parse_args()

    run_dir = ROOT / "reports" / "ultra_benchmark_sol" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"run folder not found: {run_dir}")

    summary_path = run_dir / "summary_all.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing summary_all.csv: {summary_path}")

    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(summary_path)
    bench = _to_num(bench, ["trades", "sharpe", "profit_factor", "calmar", "max_drawdown", "expectancy", "recovery_factor"])

    specs = {s.name: s for s in registry.list()}

    rows = []
    for _, r in bench.iterrows():
        name = str(r.get("strategy_name", ""))
        spec = specs.get(name)
        forbidden_refs = spec.forbidden_refs if spec else []

        leakage_warning_present = str(r.get("leakage_warnings", "[]")).strip() not in {"", "[]", "nan", "NaN"}
        code_leakage_suspect = len(forbidden_refs) > 0

        sharpe = r.get("sharpe", np.nan)
        pf = r.get("profit_factor", np.nan)
        calmar = r.get("calmar", np.nan)
        maxdd = r.get("max_drawdown", np.nan)
        trades = r.get("trades", np.nan)

        nan_or_inf_metrics = any([
            not np.isfinite(sharpe) if pd.notna(sharpe) else True,
            not np.isfinite(pf) if pd.notna(pf) else True,
            not np.isfinite(calmar) if pd.notna(calmar) else True,
            not np.isfinite(maxdd) if pd.notna(maxdd) else True,
        ])
        zero_trades = (pd.isna(trades) or float(trades) <= 0)
        too_few_trades = (not pd.isna(trades)) and float(trades) < args.min_trades
        metric_anomaly = (
            (pd.notna(sharpe) and abs(float(sharpe)) > args.sharpe_anomaly)
            or (pd.notna(calmar) and abs(float(calmar)) > args.calmar_anomaly)
        )
        maxdd_too_high = (pd.notna(maxdd) and float(maxdd) > args.maxdd_threshold)

        outer_folds_txt = str(r.get("outer_folds", ""))
        perfect_auc_suspect = ("'oos_auc': 1.0" in outer_folds_txt) or ('"oos_auc": 1.0' in outer_folds_txt)

        reasons = []
        if code_leakage_suspect:
            reasons.append("forbidden_column_reference")
        if leakage_warning_present:
            reasons.append("leakage_warning")
        if nan_or_inf_metrics:
            reasons.append("nan_or_inf_metrics")
        if zero_trades:
            reasons.append("zero_trades")
        if metric_anomaly:
            reasons.append("metric_anomaly")
        if maxdd_too_high:
            reasons.append("maxdd_too_high")
        if perfect_auc_suspect:
            reasons.append("perfect_auc_suspect")

        excluded = len(reasons) > 0

        rows.append({
            "strategy_name": name,
            "strategy_family": r.get("strategy_family", ""),
            "strategy_version": r.get("strategy_version", ""),
            "forbidden_refs": ",".join(forbidden_refs),
            "leakage_warning_present": leakage_warning_present,
            "code_leakage_suspect": code_leakage_suspect,
            "nan_or_inf_metrics": nan_or_inf_metrics,
            "zero_trades": zero_trades,
            "too_few_trades": bool(too_few_trades),
            "metric_anomaly": metric_anomaly,
            "maxdd_too_high": maxdd_too_high,
            "perfect_auc_suspect": perfect_auc_suspect,
            "excluded": excluded,
            "exclusion_reason": ";".join(reasons),
            "trades": trades,
            "sharpe": sharpe,
            "profit_factor": pf,
            "calmar": calmar,
            "max_drawdown": maxdd,
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "reality_audit.csv", index=False)

    realistic = audit[~audit["excluded"]].copy()
    realistic = realistic.sort_values(["sharpe", "profit_factor", "calmar"], ascending=False)
    realistic.to_csv(out_dir / "summary_realistic.csv", index=False)
    realistic.head(10).to_csv(out_dir / "summary_realistic_top10.csv", index=False)

    # Portfolio estimate on realistic top10 via equity endpoints when available
    top10_names = realistic.head(10)["strategy_name"].tolist()
    total_final = 0.0
    for n in top10_names:
        ep = run_dir / n / "equity.csv"
        if ep.exists():
            e = pd.read_csv(ep)
            if len(e) > 0 and "equity" in e.columns:
                total_final += float(e["equity"].iloc[-1])
            else:
                total_final += 10000.0
        else:
            total_final += 10000.0
    portfolio_return = (total_final / (10000.0 * max(len(top10_names), 1))) - 1.0

    # buy&hold reference (same horizon)
    date_from = str(bench["date_from"].iloc[0])
    date_to = str(bench["date_to"].iloc[0])
    market_csv = ROOT / "data" / "market" / "binance_spot" / "SOLUSDT" / "SOLUSDT_1h_2020-01-15_2028-12-07.csv"
    bh_final_10k = np.nan
    if market_csv.exists():
        m = pd.read_csv(market_csv)
        m["ts"] = pd.to_datetime(m["ts"], utc=True)
        sub = m[(m["ts"] >= pd.Timestamp(date_from, tz="UTC")) & (m["ts"] < pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1))]
        if len(sub) >= 2:
            ret = float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0)
            bh_final_10k = 10000.0 * (1.0 + ret)

    notes = f"""# Reality filter — {args.run_id}

## Objectif
Nettoyer le benchmark pour obtenir un classement exploitable en supprimant les stratégies leakées/suspectes et les métriques aberrantes.

## Règles appliquées
- exclusion si référence code à colonnes interdites (label/label_tb/label_fh/event_end_idx)
- exclusion si leakage warning reporté
- exclusion si métriques NaN/inf
- exclusion si trades <= 0
- exclusion si anomalie métriques (|Sharpe| > {args.sharpe_anomaly} ou |Calmar| > {args.calmar_anomaly})
- exclusion si max_drawdown > {args.maxdd_threshold}

## Résumé
- stratégies auditées: {len(audit)}
- stratégies exclues: {int(audit['excluded'].sum())}
- stratégies retenues: {int((~audit['excluded']).sum())}
- top10 réaliste disponible: summary_realistic_top10.csv

## Portefeuille (top10 réaliste)
- final estimé (10k/stratégie): {total_final:,.2f} USD
- return moyen panier: {portfolio_return*100:.2f}%
- buy&hold 10k (même horizon): {bh_final_10k:,.2f} USD
"""
    (out_dir / "reality_notes.md").write_text(notes)

    print(json.dumps({
        "run_id": args.run_id,
        "audited": int(len(audit)),
        "excluded": int(audit["excluded"].sum()),
        "retained": int((~audit["excluded"]).sum()),
        "output_dir": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
