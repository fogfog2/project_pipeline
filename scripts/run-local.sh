#!/usr/bin/env bash
set -euo pipefail

# Use a project-specific API port by default; override API_PORT/UI_PORT when
# another local service already owns either port.
API_PORT="${API_PORT:-8010}"
UI_PORT="${UI_PORT:-5173}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/uvicorn" ]]; then
  echo "Missing .venv/bin/uvicorn. Create the virtualenv and run pip install -e . first." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Vision Lifecycle API: http://127.0.0.1:${API_PORT}"
echo "Vision Lifecycle UI : http://127.0.0.1:${UI_PORT}"
VISION_LIFECYCLE_EXTERNAL_WORKER="${VISION_LIFECYCLE_EXTERNAL_WORKER:-false}" \
  "$PROJECT_ROOT/.venv/bin/uvicorn" vision_lifecycle.main:app --app-dir "$PROJECT_ROOT/backend" --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

cd "$PROJECT_ROOT/frontend"
VITE_API_PORT="$API_PORT" npm run dev -- --host 127.0.0.1 --port "$UI_PORT"
