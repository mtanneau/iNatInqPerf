#!/usr/bin/env bash
set -euo pipefail
docker rm -f qdrant || true
echo "Qdrant stopped"