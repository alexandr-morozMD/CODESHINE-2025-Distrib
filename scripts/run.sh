#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "[run] Missing .venv. Run scripts/setup.sh first." >&2
  exit 1
fi

OLLAMA_STARTED=0
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "[run] Starting ollama serve"
  ollama serve > ollama.log 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED=1
  trap 'if [ "$OLLAMA_STARTED" -eq 1 ] && kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then kill "$OLLAMA_PID"; fi' EXIT
  sleep 2
else
  echo "[run] ollama serve already running"
fi

. .venv/bin/activate

export OLLAMA_MODEL="${OLLAMA_MODEL:-phi3:mini}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

yes '' | streamlit run app.py
