#!/usr/bin/env bash
set -euo pipefail
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
echo "Qdrant started on localhost:6333"