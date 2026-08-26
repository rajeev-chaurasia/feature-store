"""The online serving API: one endpoint, thin HTTP plumbing over ``online.store``.

All the actual work, pipelining, decoding, rollup, freshness, lives in
``asofline.online.store.OnlineStore``. This module's only job is to shape an HTTP request
into a ``FeatureView`` plus entity dicts, call the store once, and shape the result back.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from asofline.config import SETTINGS
from asofline.definitions.errors import DefinitionError
from asofline.demo.views import DEMO_REGISTRY
from asofline.online.store import OnlineStore


class OnlineFeaturesRequest(BaseModel):
    view: str
    entities: list[dict[str, str]] = Field(default_factory=list)
    as_of_ms: int | None = None


class OnlineFeaturesResponse(BaseModel):
    as_of_ms: int
    results: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.store = OnlineStore.from_url(SETTINGS.redis_url)
    try:
        yield
    finally:
        await app.state.store.close()


app = FastAPI(title="asofline online serving", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/get-online-features", response_model=OnlineFeaturesResponse)
async def get_online_features(request: OnlineFeaturesRequest) -> OnlineFeaturesResponse:
    try:
        view = DEMO_REGISTRY.view(request.view)
    except DefinitionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    as_of_ms = request.as_of_ms if request.as_of_ms is not None else int(time.time() * 1000)
    store: OnlineStore = app.state.store
    try:
        vectors = await store.get_online_features(view, request.entities, as_of_ms=as_of_ms)
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    results = [
        {**entity, "features": features}
        for entity, features in zip(request.entities, vectors, strict=True)
    ]
    return OnlineFeaturesResponse(as_of_ms=as_of_ms, results=results)
