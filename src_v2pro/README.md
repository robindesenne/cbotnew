# src_v2pro (Phase 1)

Phase 1 delivered:
- isolated architecture (`src_v2pro/*`)
- strict feature allowlist (anti-leakage)
- calibrated model + threshold chosen on calibration only
- purged walk-forward scaffolding with fallback split OOS
- execution through conservative simulator

Entrypoint:
```bash
.venv/bin/python scripts_v2pro/run_backtest_v2pro.py --symbol SOLUSDT --interval 4h --from 2022-01-01 --to 2026-03-29 --cash 10000 --export reports/v2pro_phase1_sol_4h
```
