# Live Trading Rollout (V1) — Phases ultra contrôlées

Objectif: passer de backtest -> paper -> live réel de façon sécurisée sur SOLUSDT 15m (capital initial 1000 USDT).

## Phase 0 — Prérequis (NO TRADE)
- Compte Binance prêt
- Clés API créées (spot trading ON, withdrawal OFF)
- IP whitelist activée
- Secret stocké hors git (`.env`)
- Vérif horloge serveur (NTP)

## Phase 1 — Infrastructure contrôlée (NO TRADE)
- Bot en mode `dry_run` uniquement
- Lecture bougies + génération signal
- Journalisation complète des décisions
- Heartbeat/healthcheck
- Kill switch global

Critère de sortie:
- 0 crash en 72h
- 100% décisions loguées
- Aucune action externe (pas d’ordre)

## Phase 2 — Paper Trading (SIMULÉ)
- Exécution simulée “comme le réel”
- Fees/slippage configurables
- Position state machine (flat/long)
- Reconciliation régulière avec marché

Critère de sortie:
- >= 2 semaines stables
- PnL cohérent vs attentes
- 0 bug bloquant execution state

## Phase 3 — Live micro-size
- Taille min de position (ex: 50 USDT)
- 1 seule paire, 1 seul intervalle
- Max 1 position ouverte
- Daily loss cap + max drawdown cap
- Alerting Telegram à chaque ordre

Critère de sortie:
- >= 2 semaines sans incident
- slippage/frais conformes

## Phase 4 — Montée progressive vers 1000 USDT
- Progression par paliers (100 -> 250 -> 500 -> 1000)
- validation à chaque palier
- rollback immédiat si anomalie

---

## Règles de sécurité (toujours actives)
1. `dry_run=true` par défaut
2. `enable_live_orders=false` tant que non validé explicitement
3. kill switch fichier + variable d’env
4. arrêt auto si:
   - API indisponible
   - erreur de reconciliation
   - drawdown limite atteint

## Données minimales à logger
- timestamp, candle_close_ts
- signal brut + signal final
- prix de référence
- taille calculée
- raison de non-exécution
- état position avant/après
- erreurs API
