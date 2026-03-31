#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_v2.backtest import run_backtest_v2


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["15m", "1h", "4h"]
DATE_FROM = "2022-01-01"
DATE_TO = "2026-03-29"
CASH = 10_000.0


def _metric(summary: dict) -> dict:
    return {
        "symbol": summary.get("symbol"),
        "interval": summary.get("interval"),
        "trades": summary.get("trades"),
        "win_rate": summary.get("win_rate"),
        "profit_factor": summary.get("profit_factor"),
        "return_pct": summary.get("return_pct"),
        "max_drawdown": summary.get("max_drawdown"),
        "alpha_vs_bh_pct": summary.get("alpha_vs_bh_pct"),
        "wf_windows": summary.get("wf_windows"),
        "validation_mode": summary.get("validation_mode"),
    }


def main() -> int:
    out_dir = ROOT / "reports" / "strict_audit_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in SYMBOLS:
        for iv in INTERVALS:
            try:
                summary, trades, equity = run_backtest_v2(ROOT, s, iv, DATE_FROM, DATE_TO, CASH)
                rows.append(_metric(summary))
                run_dir = out_dir / f"{s}_{iv}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                trades.to_csv(run_dir / "trades.csv", index=False)
                equity.to_csv(run_dir / "equity.csv", index=False)
            except Exception as e:
                rows.append({"symbol": s, "interval": iv, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "strict_oos_summary_v2.csv", index=False)

    verdict = "INCONCLUSIVE"
    ok = df.dropna(subset=["profit_factor", "max_drawdown"]) if not df.empty else df
    if not ok.empty:
        if (ok["profit_factor"] > 1.1).mean() >= 0.6 and ok["max_drawdown"].median() < 0.25:
            verdict = "GO_PAPER_ONLY"
        else:
            verdict = "NO_GO"

    report = {
        "verdict": verdict,
        "note": "V2 strict audit based on purged walk-forward OOS backtests.",
        "rows": len(df),
    }
    (out_dir / "audit_report_v2.json").write_text(json.dumps(report, indent=2))

    print(df.to_string(index=False))
    print("Verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
