# Live Trading Infra (phase 1)

Phase actuelle: **contrôle infra uniquement** (pas d'ordres live).

## 1) Valider la config
```bash
cd /home/ubuntu/.openclaw/workspace/crypto-bot
.venv/bin/python live/validate.py --config live/config.paper.yaml
```

## 2) Lancer le bot en paper/dry-run (phase 1, monitoring)
```bash
cd /home/ubuntu/.openclaw/workspace/crypto-bot
.venv/bin/python live/bot.py --config live/config.paper.yaml
```

## 3) Lancer le paper engine V1 (phase 2, stateful)
```bash
cd /home/ubuntu/.openclaw/workspace/crypto-bot
.venv/bin/python live/paper_v1.py --config live/config.paper.yaml
```

## 4) Arrêt d'urgence (kill switch)
```bash
touch /home/ubuntu/.openclaw/workspace/crypto-bot/live/KILL_SWITCH
```

## Logs
- `live/logs/live_bot.jsonl`

## Phase 3 — Live-ready (avec garde-fous)

### A. Préflight Binance (aucun ordre)
```bash
cd /home/ubuntu/.openclaw/workspace/crypto-bot
export BINANCE_API_KEY='...'
export BINANCE_API_SECRET='...'
.venv/bin/python live/preflight_live.py --config live/config.live.yaml
```

### B. Lancer moteur live V1 (toujours bloqué par garde-fous)
```bash
cp live/config.live.yaml.template live/config.live.yaml
# garder enable_live_orders: false pour le 1er run
.venv/bin/python live/live_v1.py --config live/config.live.yaml
```

### C. Armer explicitement le système (double garde)
Conditions obligatoires pour autoriser un ordre réel:
1. `enable_live_orders: true` dans `live/config.live.yaml`
2. fichier `live/ARMED` présent

```bash
touch /home/ubuntu/.openclaw/workspace/crypto-bot/live/ARMED
```

### D. Kill switch immédiat
```bash
touch /home/ubuntu/.openclaw/workspace/crypto-bot/live/KILL_SWITCH
```

### E. Scripts de démarrage/arrêt sécurisés (chargent les secrets automatiquement)
```bash
# 1) Préparer les secrets hors repo
mkdir -p /home/ubuntu/.secrets
cp /home/ubuntu/.openclaw/workspace/crypto-bot/live/.env.example /home/ubuntu/.secrets/crypto-bot.env
nano /home/ubuntu/.secrets/crypto-bot.env
chmod 600 /home/ubuntu/.secrets/crypto-bot.env

# 2) Lancer / stopper / status
bash /home/ubuntu/.openclaw/workspace/crypto-bot/live/start_live_v1.sh
bash /home/ubuntu/.openclaw/workspace/crypto-bot/live/status_live_v1.sh
bash /home/ubuntu/.openclaw/workspace/crypto-bot/live/stop_live_v1.sh
```

## Notes sécurité
- `mode: paper` + `enable_live_orders: false` par défaut
- Le moteur live refuse d'envoyer des ordres sans double garde (`enable_live_orders` + `ARMED`)
- Toujours tester d’abord avec `test_quote: 20.0` (quote asset du symbole, ici USDC)
- Pour éviter tout risque de sur-exposition: définir aussi `hard_max_quote_per_order: 20.0`
- Optionnel: `max_use_of_free_quote_pct` pour plafonner en % du solde libre
