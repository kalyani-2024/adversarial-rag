#!/bin/sh
# Container entrypoint. APP_ROLE selects what runs:
#   api (default) - FastAPI on $PORT (default 8000)
#   ui            - Streamlit on $PORT (default 8501); set API_BASE_URL
#   all           - both in one container, UI public on $PORT (default 7860); for Hugging Face Spaces
set -eu
case "${APP_ROLE:-api}" in
  api) exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  ui)  exec streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 ;;
  all) exec sh scripts/start_all.sh ;;
  *)   echo "Unknown APP_ROLE '${APP_ROLE}' (expected api|ui|all)" >&2; exit 2 ;;
esac
