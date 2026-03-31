# Phase 2-8 — Nouveau framework (implémentation)

## Architecture implémentée

- `src/data_loader.py`
- `src/features.py`
- `src/labels.py`
- `src/regime.py`
- `src/models.py`
- `src/signals.py`
- `src/execution.py`
- `src/backtest.py`
- `src/walkforward.py`
- `src/reporting.py`
- `scripts/run_backtest.py`
- `scripts/run_walkforward.py`
- `scripts/train_model.py`
- `scripts/evaluate_model.py`

## Méthodologie

1. **Régime**: règles robustes (bull/bear/range) + transition markov empirique.
2. **Setups primaires**: trend pullback, breakout+expansion, continuation post-compression.
3. **Meta-modèle**: Logistic/RF/HGB + calibration isotonic + seuil probabiliste.
4. **Labels**: triple barrier + fixed horizon fallback.
5. **Validation**: split chrono et walk-forward (fenêtres glissantes).
6. **Exécution**: next-bar open, frais/slippage, stop/target conservateur, sizing cash/risk-based.

## Backtest réaliste
- spot long-only par défaut
- no leverage caché
- max_position_pct
- min_notional
- daily_loss_limit
- cooldown après pertes
- handling conservateur stop/target même bougie

## Critères d'acceptation (à monitorer OOS)
- profit_factor > 1.10
- drawdown inférieur à la version précédente
- stabilité multi-actifs

Si non atteint: documenter et itérer hypothèses, sans sur-optimiser.
