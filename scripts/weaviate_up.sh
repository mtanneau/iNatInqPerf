#!/usr/bin/env bash
set -euo pipefail

# Start Weaviate (bare metal).
# Requires a Weaviate server binary.
#
# Binaries:
#   - Set WEAVIATE_BIN to the absolute path of the weaviate binary
#     or put the binary on your PATH as `weaviate`.
#
# Usage:
#   scripts/weaviate_up_bare.sh [PORT] [DATA_DIR] [LOG_DIR]
#
# Defaults:
#   PORT=8080
#   DATA_DIR=./data/weaviate
#   LOG_DIR=./logs
#
# Environment overrides:
#   WEAVIATE_BIN=/path/to/weaviate
#   WEAVIATE_NAME=weaviate-bare
#   AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true|false
#   QUERY_DEFAULTS_LIMIT=25
#   PERSISTENCE_DATA_PATH=<dir>
#
# Health endpoint:
#   http://localhost:<PORT>/v1/.well-known/ready

PORT="${1:-8080}"
DATA_DIR="${2:-./data/weaviate}"
LOG_DIR="${3:-./logs}"
NAME="${WEAVIATE_NAME:-weaviate-bare}"
BIN="${WEAVIATE_BIN:-$(command -v weaviate || true)}"

if [[ -z "${BIN}" ]]; then
  echo "ERROR: Weaviate binary not found. Set WEAVIATE_BIN or install 'weaviate' on PATH." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${LOG_DIR}" .run

# Allow env overrides but provide sane defaults
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="${AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED:-true}"
export QUERY_DEFAULTS_LIMIT="${QUERY_DEFAULTS_LIMIT:-25}"
export PERSISTENCE_DATA_PATH="${PERSISTENCE_DATA_PATH:-${DATA_DIR}}"

PID_FILE=".run/${NAME}.pid"
LOG_FILE="${LOG_DIR}/${NAME}.log"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Weaviate appears to be running (pid $(cat "${PID_FILE}"))."
  exit 0
fi

echo "Starting Weaviate on port ${PORT} (data: ${DATA_DIR}) ..."
# Many builds bind to 0.0.0.0:8080 by default; if your binary supports flags,
# add them here (e.g., --host 0.0.0.0 --port ${PORT}). If not, it will still listen on 8080.
# We rely on env vars for config.
nohup "${BIN}" \
  >"${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

# Simple health check loop
echo -n "Waiting for Weaviate to become READY"
for i in {1..40}; do
  if curl -fsS "http://localhost:${PORT}/v1/.well-known/ready" >/dev/null 2>&1; then
    echo -e "\nWeaviate is READY at http://localhost:${PORT}"
    echo "Logs: ${LOG_FILE}"
    exit 0
  fi
  echo -n "."
  sleep 0.5
done

echo -e "\nWARN: Weaviate did not report READY; check logs: ${LOG_FILE}"
exit 1