#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/.openclaw/workspace/crypto-bot"
SECRETS_FILE="/home/ubuntu/.secrets/crypto-bot.env"
CFG="${ROOT}/live/config.live.yaml"
CFG_TEMPLATE="${ROOT}/live/config.live.yaml.template"
LOG_DIR="${ROOT}/live/logs"
RUN_LOG="${LOG_DIR}/live.launch.log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "[ERROR] Secrets file not found: ${SECRETS_FILE}" | tee -a "${RUN_LOG}"
  exit 1
fi

perm=$(stat -c "%a" "${SECRETS_FILE}" || echo "")
if [[ "${perm}" != "600" ]]; then
  echo "[WARN] ${SECRETS_FILE} permissions are ${perm} (recommended: 600)" | tee -a "${RUN_LOG}"
fi

set -a
source "${SECRETS_FILE}"
set +a

if [[ -z "${BINANCE_API_KEY:-}" || -z "${BINANCE_API_SECRET:-}" ]]; then
  echo "[ERROR] BINANCE_API_KEY / BINANCE_API_SECRET missing after loading ${SECRETS_FILE}" | tee -a "${RUN_LOG}"
  exit 1
fi

cd "${ROOT}"

if [[ ! -f "${CFG}" ]]; then
  if [[ -f "${CFG_TEMPLATE}" ]]; then
    cp "${CFG_TEMPLATE}" "${CFG}"
    echo "[INFO] created ${CFG} from template" | tee -a "${RUN_LOG}"
  else
    echo "[ERROR] missing config file and template: ${CFG} / ${CFG_TEMPLATE}" | tee -a "${RUN_LOG}"
    exit 1
  fi
fi

# Optional safety preflight before launch
"${ROOT}/.venv/bin/python" live/preflight_live.py --config "${CFG}" | tee -a "${RUN_LOG}"

# Start live bot if not already running
if pgrep -af "live/live_v1.py --config live/config.live.yaml" >/dev/null; then
  echo "[INFO] live_v1 already running" | tee -a "${RUN_LOG}"
  exit 0
fi

nohup "${ROOT}/.venv/bin/python" live/live_v1.py --config live/config.live.yaml >> "${RUN_LOG}" 2>&1 &
echo "[OK] live_v1 started (pid $!)" | tee -a "${RUN_LOG}"
