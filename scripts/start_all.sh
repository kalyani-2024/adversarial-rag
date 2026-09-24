#!/bin/sh
# Run API + UI in one container for single-port hosts (e.g. Hugging Face Spaces).
# The API listens only on localhost; the UI is the public port ($PORT, default 7860).
# If either process exits, the container exits so the platform restarts it.
set -eu

PORT="${PORT:-7860}"

# Hugging Face Spaces serve the app inside an iframe on another domain, where Streamlit's XSRF
# cookie is not sent back, so file uploads fail with HTTP 403. Default the protection off in
# this single-container mode (override by setting the variable), and use APP_PASSWORD to gate access.
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:-false}"

uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait for the API (model warm-up + index self-heal) before exposing the UI.
python - <<'EOF'
import time, urllib.request
for _ in range(180):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).status == 200:
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("API did not become healthy")
EOF

streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 &
UI_PID=$!

# Exit as soon as either child exits.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
  sleep 2
done
kill "$API_PID" "$UI_PID" 2>/dev/null || true
exit 1
