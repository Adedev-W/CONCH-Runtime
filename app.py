"""FastAPI application for CONCH RunPod Load Balancing workers."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from conch.inference import get_runtime
from conch.inference.service import prepare_request


LOGGER = logging.getLogger("conch.api")
DEFAULT_MAX_CONCURRENCY = 1


class RankRequest(BaseModel):
    """Public request schema for one image and its candidate text queries."""

    image_base64: str
    queries: list[str]
    top_k: int | None = None


class RankResult(BaseModel):
    rank: int
    query: str
    score: float
    probability: float


class RankResponse(BaseModel):
    model: str
    results: list[RankResult]


def _max_concurrency() -> int:
    value = int(os.getenv("CONCH_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
    if value < 1:
        raise ValueError("CONCH_MAX_CONCURRENCY must be greater than zero")
    return value


async def _warm_runtime(app: FastAPI) -> None:
    try:
        # Model construction is synchronous and GPU-heavy, so keep it off the
        # event loop while allowing /ping to report the initializing state.
        await asyncio.to_thread(get_runtime)
    except Exception as exc:
        app.state.runtime_error = exc
        LOGGER.exception("CONCH runtime initialization failed")
    else:
        app.state.runtime_ready = True
        LOGGER.info("CONCH runtime is ready")


def _rank_with_loaded_runtime(image, queries, top_k):
    """Keep the singleton lookup inside the worker thread with PyTorch work."""
    return get_runtime().rank(image, queries, top_k=top_k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime_ready = False
    app.state.runtime_error = None
    app.state.inference_semaphore = asyncio.Semaphore(_max_concurrency())
    initialization_task = asyncio.create_task(_warm_runtime(app))
    app.state.initialization_task = initialization_task

    try:
        yield
    finally:
        if not initialization_task.done():
            initialization_task.cancel()
            await asyncio.gather(initialization_task, return_exceptions=True)


app = FastAPI(
    title="CONCH Runtime",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/ping", include_in_schema=False)
async def ping() -> Response:
    """Expose RunPod's required readiness contract."""
    if app.state.runtime_error is not None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    if not app.state.runtime_ready:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return Response(status_code=status.HTTP_200_OK)


@app.post("/rank", response_model=RankResponse)
async def rank(request: RankRequest) -> RankResponse:
    """Rank the supplied text queries against one pathology image."""
    if app.state.runtime_error is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CONCH runtime failed to initialize",
            headers={"Retry-After": "10"},
        )
    if not app.state.runtime_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CONCH runtime is still initializing",
            headers={"Retry-After": "5"},
        )

    try:
        payload: dict[str, Any] = request.model_dump()
        image, queries, top_k = prepare_request(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        async with app.state.inference_semaphore:
            results = await asyncio.to_thread(
                _rank_with_loaded_runtime,
                image,
                queries,
                top_k,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("CONCH inference request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CONCH inference failed",
        ) from exc

    return RankResponse(
        model=os.getenv("CONCH_MODEL_NAME", "conch_ViT-B-16"),
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "80")),
    )
