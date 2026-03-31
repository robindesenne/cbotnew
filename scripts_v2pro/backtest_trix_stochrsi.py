#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_v2.data_loader import LoadSpec, load_ohlcv
from src_v2pro.indicator_trix_stochrsi import add_trix_stochrsi_combo


def run_bt(df: pd.DataFrame, cash0: float = 10_000.0, fee: float = 0.001, slip_bps: float = 2.0):
    slip = slip_bps / 10000.0
    cash = cash0
    qty = 0.0
    entry = None
    trades = []
    eq = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        px = float(row["close"])

        equity = cash + qty * px
        eq.append({"ts": row["ts"], "equity": equity})

        if qty <= 0 and int(row.get("combo_long", 0)) == 1:
            buy_px = float(nxt["open"]) * (1 + slip)
            qty = (cash / (buy_px * (1 + fee))) if buy_px > 0 else 0.0
            if qty > 0:
                cost = qty * buy_px * (1 + fee)
                cash -= cost
                entry = buy_px

        elif qty > 0 and int(row.get("combo_exit", 0)) == 1:
            sell_px = float(nxt["open"]) * (1 - slip)
            gross = (sell_px - float(entry)) * qty
            fees = (float(entry) * qty + sell_px * qty) * fee
            net = gross - fees
            cash += sell_px * qty * (1 - fee)
            trades.append({"entry": entry, "exit": sell_px, "qty": qty, "gross": gross, "fees": fees, "net": net, "exit_ts": str(row["ts"])})
            qty = 0.0
            entry = None

    eqdf = pd.DataFrame(eq)
    tdf = pd.DataFrame(trades)
    net = float(tdf["net"].sum()) if len(tdf) else 0.0
    wins = int((tdf["net"] > 0).sum()) if len(tdf) else 0
    gp = float(tdf.loc[tdf.net > 0, "net"].sum()) if len(tdf) else 0.0
    gl = float(-tdf.loc[tdf.net <= 0, "net"].sum()) if len(tdf) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    peak = eqdf["equity"].cummax() if len(eqdf) else pd.Series(dtype=float)
    dd = ((peak - eqdf["equity"]) / peak).max() if len(eqdf) else 0.0

    return {
        "trades": int(len(tdf)),
        "win_rate": float(wins / len(tdf)) if len(tdf) else 0.0,
        "net_pnl": net,
        "return_pct": float(net / cash0),
        "profit_factor": float(pf),
        "max_drawdown": float(dd if pd.notna(dd) else 0.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--cash", type=float, default=10000)
    p.add_argument("--out", default="reports/trix_stochrsi_bt")
    args = p.parse_args()

    df, source = load_ohlcv(ROOT, LoadSpec(symbol=args.symbol, interval=args.interval, date_from=args.date_from, date_to=args.date_to, market_type="spot"))
    df = add_trix_stochrsi_combo(df)
    metrics = run_bt(df, cash0=args.cash)
    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "source": source,
        **metrics,
    }

    out = ROOT / args.out / f"{args.symbol}_{args.interval}_{args.date_from}_{args.date_to}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
