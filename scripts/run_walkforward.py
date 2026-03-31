#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import save_dataframe
from src.walkforward import WFSpec, run_walkforward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--cash", type=float, default=10000)
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=60)
    p.add_argument("--step-days", type=int, default=60)
    p.add_argument("--out", default="reports/walkforward/results.csv")
    args = p.parse_args()

    root = ROOT
    df = run_walkforward(
        root=root,
        symbol=args.symbol,
        interval=args.interval,
        date_from=args.date_from,
        date_to=args.date_to,
        cash=args.cash,
        spec=WFSpec(args.train_days, args.test_days, args.step_days),
    )
    save_dataframe(root / args.out, df)
    print(df.tail())


if __name__ == "__main__":
    main()
