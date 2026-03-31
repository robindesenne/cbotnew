#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--interval', default='1h')
    p.add_argument('--from', dest='date_from', required=True)
    p.add_argument('--to', dest='date_to', required=True)
    p.add_argument('--cash', type=float, default=10000)
    p.add_argument('--export', default='reports/backtest_v1')
    args = p.parse_args()

    frozen = ROOT / 'versions' / 'v1_frozen' / 'scripts' / 'run_backtest.py'
    cmd = [
        str(ROOT / '.venv' / 'bin' / 'python'), str(frozen),
        '--symbol', args.symbol,
        '--interval', args.interval,
        '--from', args.date_from,
        '--to', args.date_to,
        '--cash', str(args.cash),
        '--export', args.export,
    ]
    env = dict(__import__("os").environ)
    env["V1_PROJECT_ROOT"] = str(ROOT)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
