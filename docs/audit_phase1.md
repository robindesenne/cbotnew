# Phase 1 — Audit système actuel (avant refonte)

## Constat global
Backtests négatifs structurels sur BTCUSDC / ETHUSDC / SOLUSDC, même après première refonte.
Le moteur est réaliste sur certains points (next-bar open, frais/slippage), mais l'edge des signaux est insuffisant et la séparation recherche/simulation est trop faible.

## Défauts classés par gravité

### Critique
1. **Edge signal faible / redondant**
   - Stack indicateurs initial (TRIX/RSI/StochRSI) fortement corrélée.
   - Peu de diversification des familles de setup.
2. **Couplage recherche + exécution dans un script monolithique**
   - Difficile de tester indépendamment labels/features/signaux/exécution.
3. **Validation insuffisamment robuste**
   - Pas de pipeline walk-forward strict standardisé dans la première version.

### Élevé
4. **Sensibilité forte aux frais/slippage**
   - Plusieurs périodes avec profit factor < 1 dès coûts réalistes.
5. **Fréquence de faux signaux élevée**
   - Beaucoup d'entrées en conditions de trend faible / bruit.
6. **Peu d'analyse par sous-période/régime**
   - Difficile d'identifier précisément où la stratégie casse.

### Moyen
7. **Peu d'interprétabilité du filtre**
   - Pas de méta-modèle calibré standard dans l'ancienne version.
8. **Comparatifs baselines incomplets**
   - Buy&Hold / primaire naïf pas toujours reportés systématiquement.

## Hypothèses irréalistes / fragiles identifiées
- Hypothèse implicite que la simple confluence oscillateurs suffit à survivre à tous régimes.
- Trop peu de séparation entre détection de setup et décision de prise de trade.

## Décision conserver / jeter

### À conserver
- Exécution conservative (next bar open, stop/target conservateur, frais/slippage explicites)
- Spot long-only par défaut
- Limites de risque (daily loss, cooldown, max position)

### À jeter / remplacer
- Monolithe unique `run_backtest.py` pour tout faire
- Dépendance principale au trio TRIX/RSI/StochRSI
- Validation non standardisée sans walk-forward systématique
