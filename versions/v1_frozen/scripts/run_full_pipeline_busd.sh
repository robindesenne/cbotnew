#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/.openclaw/workspace/crypto-bot"
cd "$ROOT"

source .venv/bin/activate

echo "[$(date -u +%FT%TZ)] Start USDT campaign"
python scripts/run_campaign.py

echo "[$(date -u +%FT%TZ)] Start strict audit"
python scripts/run_strict_audit.py

echo "[$(date -u +%FT%TZ)] Full pipeline done"
