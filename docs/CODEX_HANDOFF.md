# CODEX_HANDOFF.md — crypto-bot

## Objectif
Passation rapide pour reprendre le projet crypto en local (Codex) avec le contexte des dernières itérations et une procédure de reconstruction reproductible.

## État actuel (résumé)
- Branche: `main`
- Scope principal récent: benchmark SOLUSDT 1h, comparaison multi-stratégies, publication webapp.
- Référentiel 21 stratégies: `configs/retained21.txt`
- Checkpoint courant: `configs/checkpoint_latest.md`

## Changements récents importants
1. `scripts/ultra_benchmark_sol.py`
   - ajout `--strategy-file`
   - ajout `--strategies` (CSV)
   => permet de rejouer un sous-ensemble fixe de stratégies (ex: retained21).

2. `src/strategies/solusdt_strategies.py`
   - évolution des variantes momentum/mean-reversion/breakout
   - nouvelles variantes event-driven/filters

3. Publication compacte pour la webapp
   - nouveau script: `scripts/publish_run_compact.py`
   - génère pour un run:
     - `micro_results_compact.csv`
     - `buyhold_ref.txt`

4. Rebuild local guidé
   - nouveau script: `scripts/rebuild_local_state.sh`
   - enchaîne:
     1) download Binance
     2) benchmark retained21
     3) publication compacte

## Reconstruction locale (recommandée)
```bash
bash scripts/rebuild_local_state.sh \
  --from 2020-09-14 \
  --to 2025-12-27 \
  --symbol SOLUSDT \
  --interval 1h
```

### Variante rapide (sans retélécharger)
```bash
bash scripts/rebuild_local_state.sh \
  --from 2020-09-14 \
  --to 2025-12-27 \
  --symbol SOLUSDT \
  --interval 1h \
  --skip-download
```

## Commandes unitaires utiles
```bash
python3 scripts/ultra_benchmark_sol.py \
  --run-id test-retained21 \
  --max-workers 1 \
  --interval 1h \
  --date-from 2020-09-14 \
  --date-to 2025-12-27 \
  --strategy-file configs/retained21.txt

python3 scripts/publish_run_compact.py --run-id test-retained21 --cash 10000 --symbol SOLUSDT --interval 1h --date-from 2020-09-14 --date-to 2025-12-27
```

## Notes importantes
- `reports/` est ignoré par git (normal), donc les artefacts de runs ne sont pas versionnés.
- Le script `publish_run_compact.py` est le pont pour la webapp (overlay latest run).
- Les données de marché Binance sont locales (`data/`) et peuvent être régénérées.

## Fichiers à lire en priorité
- `configs/checkpoint_latest.md`
- `configs/retained21.txt`
- `scripts/rebuild_local_state.sh`
- `scripts/publish_run_compact.py`
- `scripts/ultra_benchmark_sol.py`
- `src/strategies/solusdt_strategies.py`
