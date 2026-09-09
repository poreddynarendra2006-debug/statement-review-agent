#!/bin/sh
# Start both processes in one container.
#
# uvicorn runs in the background and Streamlit in the foreground, so the
# container's lifetime follows the UI process - when Streamlit stops, the
# container stops, which is what the orchestrator expects.
#
# If the API fails to start we log it and carry on: the workspace is the
# demo surface and must not be taken down by a failure in the integration
# surface.

set -e

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

echo "[entrypoint] starting FastAPI on :${API_PORT}"
uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

# Stop the background API cleanly when the container is asked to stop.
trap 'echo "[entrypoint] shutting down"; kill "${API_PID}" 2>/dev/null || true' TERM INT

sleep 2
if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "[entrypoint] WARNING: API failed to start; continuing with the workspace only"
fi

echo "[entrypoint] starting Streamlit on :${UI_PORT}"
exec streamlit run app.py \
    --server.port "${UI_PORT}" \
    --server.address 0.0.0.0
