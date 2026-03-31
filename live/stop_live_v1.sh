#!/usr/bin/env bash
set -euo pipefail

if pgrep -af "live/live_v1.py --config live/config.live.yaml" >/dev/null; then
  pkill -f "live/live_v1.py --config live/config.live.yaml"
  echo "[OK] live_v1 stop signal sent"
else
  echo "[INFO] live_v1 not running"
fi
