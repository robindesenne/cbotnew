#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATE_FROM="2020-09-14"
DATE_TO="2025-12-27"
SYMBOL="SOLUSDT"
INTERVAL="1h"
DOWNLOAD_INTERVALS="1h 4h"
CASH="10000"
RUN_ID=""
SKIP_DOWNLOAD=0

usage() {
  cat <<'EOF'
Rebuild local pour crypto-bot (cache marché + benchmark + publication compacte).

Usage:
  bash scripts/rebuild_local_state.sh [options]

Options:
  --from YYYY-MM-DD         Date début (défaut: 2020-09-14)
  --to YYYY-MM-DD           Date fin   (défaut: 2025-12-27)
  --symbol SYMBOL           Symbole (défaut: SOLUSDT)
  --interval TF             Timeframe benchmark (défaut: 1h)
  --download-intervals "..." Intervalles download Binance (défaut: "1h 4h")
  --cash N                  Capital initial (défaut: 10000)
  --run-id ID               Run ID explicite (sinon auto)
  --skip-download           Ne retélécharge pas les candles
  -h, --help                Aide

Exemple:
  bash scripts/rebuild_local_state.sh --from 2020-09-14 --to 2025-12-27 --symbol SOLUSDT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      DATE_FROM="$2"; shift 2 ;;
    --to)
      DATE_TO="$2"; shift 2 ;;
    --symbol)
      SYMBOL="$2"; shift 2 ;;
    --interval)
      INTERVAL="$2"; shift 2 ;;
    --download-intervals)
      DOWNLOAD_INTERVALS="$2"; shift 2 ;;
    --cash)
      CASH="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --skip-download)
      SKIP_DOWNLOAD=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERR] Option inconnue: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="full21-local-${DATE_FROM//-/}-${DATE_TO//-/}"
fi

if [[ "$SKIP_DOWNLOAD" -ne 1 ]]; then
  echo "[1/3] Download Binance history ($SYMBOL | $DOWNLOAD_INTERVALS | $DATE_FROM -> $DATE_TO)"
  # shellcheck disable=SC2086
  python3 scripts/download_binance_history.py \
    --symbols "$SYMBOL" \
    --intervals $DOWNLOAD_INTERVALS \
    --from "$DATE_FROM" \
    --to "$DATE_TO"
else
  echo "[1/3] Download skipped"
fi

echo "[2/3] Run full benchmark (retained21)"
CRYPTOBOT_LEAKAGE_MODE=error python3 scripts/ultra_benchmark_sol.py \
  --run-id "$RUN_ID" \
  --max-workers 1 \
  --cash "$CASH" \
  --interval "$INTERVAL" \
  --date-from "$DATE_FROM" \
  --date-to "$DATE_TO" \
  --strategy-file configs/retained21.txt

echo "[3/3] Publish compact report for webapp"
python3 scripts/publish_run_compact.py \
  --run-id "$RUN_ID" \
  --cash "$CASH" \
  --symbol "$SYMBOL" \
  --interval "$INTERVAL" \
  --date-from "$DATE_FROM" \
  --date-to "$DATE_TO"

echo "[OK] Rebuild terminé"
echo "Run: $RUN_ID"
echo "Path: reports/ultra_benchmark_sol/$RUN_ID"
