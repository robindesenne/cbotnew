#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-ultra-sol-v1-full}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DATE_FROM="${DATE_FROM:-2020-01-15}"
DATE_TO="${DATE_TO:-2026-03-29}"
INTERVAL="${INTERVAL:-1h}"

LOG_DIR="$ROOT_DIR/reports/ultra_benchmark_sol/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
PID_FILE="$LOG_DIR/${RUN_ID}.pid"

cd "$ROOT_DIR"

nohup bash -lc "python3 -u scripts/ultra_benchmark_sol.py --run-id '$RUN_ID' --max-workers '$MAX_WORKERS' --interval '$INTERVAL' --date-from '$DATE_FROM' --date-to '$DATE_TO' --resume" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Started RUN_ID=$RUN_ID PID=$PID"
echo "Log: $LOG_FILE"
echo "Progress: $ROOT_DIR/reports/ultra_benchmark_sol/$RUN_ID/progress.json"
