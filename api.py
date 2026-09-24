"""ASGI entry point (kept at the repo root so `uvicorn api:app` keeps working).

    uvicorn api:app --port 8000
"""

from app.api.main import create_app

app = create_app()
