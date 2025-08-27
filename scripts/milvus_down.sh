#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/milvus_down.sh [CONTAINER_NAME]
#
# Default:
#   CONTAINER_NAME=${MILVUS_CONTAINER_NAME:-milvus-standalone}

NAME="${1:-${MILVUS_CONTAINER_NAME:-milvus-standalone}}"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}\$"; then
  echo "Stopping and removing container '${NAME}'…"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  echo "Done."
else
  echo "No container named '${NAME}' found. Nothing to do."
fi