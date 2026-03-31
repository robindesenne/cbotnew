from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .backtest import run_single_backtest


@dataclass
class WFSpec:
    train_days: int = 180
    test_days: int = 60
    step_days: int = 60


def run_walkforward(root: Path, symbol: str, interval: str, date_from: str, date_to: str, cash: float, spec: WFSpec):
    d0 = pd.Timestamp(date_from, tz="UTC")
    d1 = pd.Timestamp(date_to, tz="UTC")

    rows = []
    cur = d0
    while cur + timedelta(days=spec.train_days + spec.test_days) <= d1:
        tr_from = cur
        tr_to = cur + timedelta(days=spec.train_days)
        te_to = tr_to + timedelta(days=spec.test_days)

        try:
            summary, *_ = run_single_backtest(
                root=root,
                symbol=symbol,
                interval=interval,
                date_from=tr_from.date().isoformat(),
                date_to=te_to.date().isoformat(),
                cash=cash,
            )
        except Exception:
            cur += timedelta(days=spec.step_days)
            continue

        summary["wf_train_from"] = tr_from.date().isoformat()
        summary["wf_train_to"] = tr_to.date().isoformat()
        summary["wf_test_to"] = te_to.date().isoformat()
        rows.append(summary)
        cur += timedelta(days=spec.step_days)

    return pd.DataFrame(rows)
