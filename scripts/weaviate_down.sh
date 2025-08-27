#!/usr/bin/env bash
set -euo pipefail

# Stop Weaviate started by weaviate_up_bare.sh
# Usage:
#   scripts/weaviate_down_bare.sh [NAME]
# Default NAME = ${WEAVIATE_NAME:-weaviate-bare}

NAME="${1:-${WEAVIATE_NAME:-weaviate-bare}}"
PID_FILE=".run/${NAME}.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Stopping Weaviate (pid ${PID})..."
    kill "${PID}" || true
    sleep 1
    if kill -0 "${PID}" 2>/dev/null; then
      echo "Force killing Weaviate..."
      kill -9 "${PID}" || true
    fi
  else
    echo "No running process for PID ${PID}."
  fi
  rm -f "${PID_FILE}"
else
  echo "No PID file at ${PID_FILE}; nothing to stop."
fi