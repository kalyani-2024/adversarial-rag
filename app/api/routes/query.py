from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_container
from app.core.errors import AppError
from app.schemas.query import QueryRequest, QueryResponse
from app.services.container import Container

router = APIRouter(tags=["query"])
logger = logging.getLogger("rag.api")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question with hybrid retrieval and the adversarial reliability loop",
    responses={429: {"description": "LLM rate limited"}, 502: {"description": "LLM error"},
               504: {"description": "LLM timeout"}},
)
def query(body: QueryRequest, request: Request, container: Container = Depends(get_container)):
    return container.queries.run(body, request_id=request.state.request_id)


@router.post(
    "/query/stream",
    summary="Same as /query, streamed as Server-Sent Events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": (
        "Events: `status` {stage: retrieving|generating|verifying|revising}, `sources` {sources:[{index,label,text}]}, "
        "`token` {text}, `reset` (discard the streamed draft; a revised answer follows), "
        "`final` (a full QueryResponse; authoritative answer + citations), `error` {code, message, status}."
    )}},
)
def query_stream(body: QueryRequest, request: Request, container: Container = Depends(get_container)):
    """Stream the draft answer as it is generated, then the verified final answer.

    The pipeline runs in a worker thread and pushes events onto a queue that the
    response drains. The draft shown during streaming may be replaced: if the
    judge fails it, a `reset` event is sent and the revised answer streams in.
    The `final` event always carries the answer the pipeline actually selected.
    """
    events: queue.Queue = queue.Queue()
    request_id = request.state.request_id

    def emit(event: str, data: dict) -> None:
        events.put((event, data))

    def worker() -> None:
        try:
            response = container.queries.run(body, request_id=request_id, emit=emit)
            events.put(("final", response.model_dump(mode="json")))
        except AppError as exc:
            events.put(("error", {"code": exc.code, "message": exc.message, "status": exc.status_code}))
        except Exception:
            logger.exception("streaming query failed", extra={"request_id": request_id})
            events.put(("error", {"code": "internal_error", "message": "Internal server error.", "status": 500}))
        finally:
            events.put(None)

    threading.Thread(target=worker, name=f"query-{request_id}", daemon=True).start()

    def event_stream():
        while (item := events.get()) is not None:
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # disable proxy buffering
    )
