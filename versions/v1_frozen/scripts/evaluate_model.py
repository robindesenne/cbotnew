#!/usr/bin/env python3
"""Evaluation entrypoint placeholder.
Use reports/backtest/summary.json and reports/walkforward/results.csv.
"""

from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
summary = root / "reports" / "backtest" / "summary.json"
if summary.exists():
    print(json.loads(summary.read_text()))
else:
    print("No backtest summary found. Run scripts/run_backtest.py first.")
