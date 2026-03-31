#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_single_backtest, save_backtest_outputs
from src.data_loader import LoadSpec, load_ohlcv
from src.walkforward import WFSpec, run_walkforward

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["15m", "1h", "4h"]
DATE_FROM = "2022-01-01"
DATE_TO = "2026-03-21"
CASH = 10_000.0


def buy_and_hold_return(root: Path, symbol: str, interval: str, date_from: str, date_to: str, cash: float) -> dict:
    spec = LoadSpec(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to, market_type="spot")
    df, source = load_ohlcv(root, spec)
    if len(df) < 2:
        return {"symbol": symbol, "interval": interval, "bh_return_pct": 0.0, "bh_net_pnl": 0.0, "bh_source": source}
    entry = float(df.iloc[0]["close"])
    exit_ = float(df.iloc[-1]["close"])
    # conservative roundtrip taker fees + slippage proxy
    total_cost = 0.001 * 2 + 0.0002  # approx 0.22%
    gross_ret = (exit_ / entry) - 1.0
    net_ret = gross_ret - total_cost
    return {
        "symbol": symbol,
        "interval": interval,
        "bh_return_pct": float(net_ret),
        "bh_net_pnl": float(cash * net_ret),
        "bh_source": source,
    }


def main():
    root = ROOT
    campaign_dir = root / "reports" / "campaign_2022_2026"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sym in SYMBOLS:
        for tf in INTERVALS:
            print(f"=== Backtest {sym} {tf} ===")
            run_name = f"{sym}_{tf}".replace("/", "_")
            out_dir = campaign_dir / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            summary_path = out_dir / "summary.json"
            wf_path = out_dir / "walkforward.csv"

            if summary_path.exists() and wf_path.exists():
                print(f"Skipping {sym} {tf} (already completed)")
                summary = json.loads(summary_path.read_text())
                wf = pd.read_csv(wf_path)
            else:
                summary, trades, equity, _ = run_single_backtest(
                    root=root,
                    symbol=sym,
                    interval=tf,
                    date_from=DATE_FROM,
                    date_to=DATE_TO,
                    cash=CASH,
                )
                save_backtest_outputs(root, str(out_dir.relative_to(root)), summary, trades, equity)

                wf = run_walkforward(
                    root=root,
                    symbol=sym,
                    interval=tf,
                    date_from=DATE_FROM,
                    date_to=DATE_TO,
                    cash=CASH,
                    spec=WFSpec(train_days=180, test_days=60, step_days=60),
                )
                wf.to_csv(wf_path, index=False)

            bh = buy_and_hold_return(root, sym, tf, DATE_FROM, DATE_TO, CASH)

            row = {
                **summary,
                **bh,
                "wf_windows": int(len(wf)),
                "wf_pf_median": float(wf["profit_factor"].replace([float("inf")], pd.NA).dropna().median()) if len(wf) else None,
                "wf_dd_median": float(wf["max_drawdown"].median()) if len(wf) else None,
                "alpha_vs_bh_pct": float(summary.get("return_pct", 0.0) - bh.get("bh_return_pct", 0.0)),
            }
            rows.append(row)

    report = pd.DataFrame(rows)
    report = report.sort_values(["symbol", "interval"]).reset_index(drop=True)
    report.to_csv(campaign_dir / "summary_campaign.csv", index=False)
    (campaign_dir / "summary_campaign.json").write_text(json.dumps(rows, indent=2))

    # aggregate view
    agg = {
        "runs": int(len(report)),
        "pf_median": float(report["profit_factor"].replace([float("inf")], pd.NA).dropna().median()),
        "dd_median": float(report["max_drawdown"].median()),
        "return_pct_median": float(report["return_pct"].median()),
        "alpha_vs_bh_median": float(report["alpha_vs_bh_pct"].median()),
        "wins_pf_gt_1_10": int((report["profit_factor"] > 1.10).sum()),
    }
    (campaign_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    print("Campaign done")
    print(report[["symbol", "interval", "model", "profit_factor", "max_drawdown", "return_pct", "bh_return_pct", "alpha_vs_bh_pct"]])
    print("Aggregate:", agg)


if __name__ == "__main__":
    main()
