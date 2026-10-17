#!/bin/bash
# Build the site and serve the production bundle over HTTPS on :8655 (inside the isobox container).
set -e
cd "$(dirname "$0")/../web"
for p in $(pgrep -f "vite.*--port 865[5]" || true); do kill "$p" 2>/dev/null || true; done
HTTPS=1 npx vite build 2>&1 | grep -E "built in|error" || true
HTTPS=1 nohup npx vite preview --port 8655 --strictPort > /tmp/serve.log 2>&1 &
for i in $(seq 1 40); do curl -skf -o /dev/null https://localhost:8655/ && break; sleep 0.5; done
curl -sk -o /dev/null -w "https://localhost:8655 -> %{http_code}\n" https://localhost:8655/
