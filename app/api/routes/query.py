from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_container
from app.schemas.query import QueryRequest, QueryResponse
from app.services.container import Container

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question with hybrid retrieval and the adversarial reliability loop",
    responses={429: {"description": "LLM rate limited"}, 502: {"description": "LLM error"},
               504: {"description": "LLM timeout"}},
)
def query(body: QueryRequest, request: Request, container: Container = Depends(get_container)):
    return container.queries.run(body, request_id=request.state.request_id)
