#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/.openclaw/workspace/crypto-bot"
LOG_JSONL="${ROOT}/live/logs/live_bot.live.jsonl"

if pgrep -af "live/live_v1.py --config live/config.live.yaml" >/dev/null; then
  echo "STATUS: RUNNING"
  pgrep -af "live/live_v1.py --config live/config.live.yaml"
else
  echo "STATUS: STOPPED"
fi

echo "--- tail ${LOG_JSONL}"
if [[ -f "${LOG_JSONL}" ]]; then
  tail -n 30 "${LOG_JSONL}"
else
  echo "(log not found)"
fi
