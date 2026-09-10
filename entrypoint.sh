#!/bin/sh
# Start FinSight AI.
#
# Container hosts pass the port to listen on through $PORT; default 8000.
# exec makes uvicorn process 1, so it receives the host's stop signal directly
# and shuts down cleanly instead of being killed.

set -e

PORT="${PORT:-8000}"

echo "[entrypoint] starting FinSight AI on :${PORT}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --proxy-headers
