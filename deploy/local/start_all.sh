#!/usr/bin/env bash
# start_all.sh — Start all 3 Streamlit apps + Caddy (or Nginx) + ngrok
#
# Usage:
#   ./deploy/local/start_all.sh          # uses Caddy by default
#   PROXY=nginx ./deploy/local/start_all.sh
#
# Requirements:
#   brew install caddy   (or nginx)
#   brew install ngrok
#   pip install streamlit (in each project venv)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROXY="${PROXY:-caddy}"

echo "==> Starting from: $REPO_ROOT"

# ---- App 1: Enterprise RAG ----
echo "==> Starting App 1 (Enterprise RAG) on :8501…"
cd "$REPO_ROOT"
./.venv/bin/streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.baseUrlPath app1 \
  --server.headless true &
APP1_PID=$!

# ---- App 2 & 3: placeholders ----
# Uncomment and update paths when ready:
#
# echo "==> Starting App 2 on :8502…"
# cd /path/to/project2
# ./.venv/bin/streamlit run app.py \
#   --server.port 8502 \
#   --server.baseUrlPath app2 \
#   --server.headless true &
# APP2_PID=$!
#
# echo "==> Starting App 3 on :8503…"
# cd /path/to/project3
# ./.venv/bin/streamlit run app.py \
#   --server.port 8503 \
#   --server.baseUrlPath app3 \
#   --server.headless true &
# APP3_PID=$!

sleep 2  # give Streamlit a moment to bind ports

# ---- Proxy ----
echo "==> Starting proxy ($PROXY) on :8080…"
cd "$REPO_ROOT"
if [[ "$PROXY" == "nginx" ]]; then
  nginx -c "$(pwd)/deploy/local/nginx.conf" -p "$(pwd)/deploy/local"
  PROXY_CMD="nginx -c $(pwd)/deploy/local/nginx.conf -p $(pwd)/deploy/local -s stop"
else
  caddy start --config deploy/local/Caddyfile
  PROXY_CMD="caddy stop"
fi

# ---- ngrok ----
echo "==> Starting ngrok on :8080…"
ngrok http 8080 &
NGROK_PID=$!

echo ""
echo "All services running. Visit the ngrok URL printed above."
echo "Press Ctrl+C to stop everything."
echo ""

# ---- Graceful shutdown on Ctrl+C ----
cleanup() {
  echo ""
  echo "==> Shutting down…"
  kill "$APP1_PID" 2>/dev/null || true
  # kill "$APP2_PID" 2>/dev/null || true
  # kill "$APP3_PID" 2>/dev/null || true
  kill "$NGROK_PID" 2>/dev/null || true
  eval "$PROXY_CMD" 2>/dev/null || true
  echo "Done."
}
trap cleanup INT TERM

wait
