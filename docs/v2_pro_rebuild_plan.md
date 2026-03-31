# V2 Pro Rebuild Plan (anti-overfitting)

## Objective
Rebuild V2 from scratch with institutional-grade validation and execution realism.

## Non-negotiables
- No label leakage
- Purged + embargoed walk-forward only
- Nested CV for threshold/model selection
- Out-of-sample reporting only
- Cost/slippage/capacity stress required
- Paper-trading gate before live

## Phase 1 — Clean Architecture
- New isolated package: `src_v2pro/`
- Strict feature allowlist (no derived target columns)
- Single source of truth config: `config_v2_pro.yaml`

## Phase 2 — Data & Feature Pipeline
- Deterministic feature generation (OHLCV only first)
- Timestamp-safe joins only (`asof`, no future merge)
- Feature QA: NaN leakage checks + stationarity diagnostics

## Phase 3 — Signal & Modeling
- Base alpha blocks:
  1. Momentum continuation
  2. Volatility compression breakout
  3. Mean reversion veto (risk filter)
- Meta-model: calibrated probabilistic classifier
- Regime-aware activation (trend/chop/crash)

## Phase 4 — Validation (core)
- Rolling windows: train/cal/test
- Purge overlapping labels + embargo around test
- Threshold chosen on calibration only
- Test block strictly untouched until final scoring

## Phase 5 — Execution realism
- Fees + slippage model by interval/volatility
- Min notional/step size constraints
- Partial-fill conservative approximation
- Capacity cap (max % ADV proxy)

## Phase 6 — Audit pack
- Placebo tests: shuffled labels / feature permutation
- Stress: fee x2, slip x2, latency delay
- Robustness: per-year, per-regime, per-symbol
- Decision: GO_PAPER_ONLY or NO_GO

## Success criteria
- PF > 1.15 median OOS
- Max DD < 20% median OOS
- Stable across symbols/years
- Placebo collapses to noise
