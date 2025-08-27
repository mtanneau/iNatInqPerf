#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/milvus_up.sh [PORT] [DATA_DIR] [IMAGE]
#
# Defaults:
#   PORT=19530
#   DATA_DIR=./data/milvus
#   IMAGE=${MILVUS_IMAGE:-milvusdb/milvus:v2.4.6-standalone}

PORT="${1:-19530}"
DATA_DIR="${2:-./data/milvus}"
IMAGE="${3:-${MILVUS_IMAGE:-milvusdb/milvus:v2.4.6-standalone}}"
NAME="${MILVUS_CONTAINER_NAME:-milvus-standalone}"

mkdir -p "$DATA_DIR"

# Stop/remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}\$"; then
  echo "Container '${NAME}' already exists. Removing…"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
fi

echo "Starting Milvus '${IMAGE}' on port ${PORT} with data dir '${DATA_DIR}' (container: ${NAME})"
docker run -d \
  --name "${NAME}" \
  -p "${PORT}:19530" \
  -p 9091:9091 \
  -e ETCD_USE_EMBED=true \
  -e MINIO_USE_EMBED=true \
  -e PULSAR_USE_EMBED=true \
  -v "${DATA_DIR}:/var/lib/milvus" \
  "${IMAGE}"

echo "Milvus is up: port ${PORT}"
echo "Health check: curl -s http://localhost:9091/healthz"