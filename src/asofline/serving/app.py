"""The online serving API: one endpoint, thin HTTP plumbing over ``online.store``.

All the actual work, pipelining, decoding, rollup, freshness, lives in
``asofline.online.store.OnlineStore``. This module's only job is to shape an HTTP request
into a ``FeatureView`` plus entity dicts, call the store once, and shape the result back.

**Feature logging is fire-and-forget, on purpose.** The P5 skew detector needs to know
what was actually served, but a request that has already computed its answer must not
wait on a Kafka publish before returning it: logging is scheduled as a background task
that the response does not await, and any exception it raises is caught and counted rather
than propagated, because a broker hiccup is not a reason to fail a feature request.
``feature_log_sample_rate`` exists because logging every request at production volume is
not free; it defaults to 1.0 (log everything) so tests and small demos see every entry
without needing to know the knob exists.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from random import Random
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from asofline.config import SETTINGS
from asofline.definitions.errors import DefinitionError
from asofline.definitions.view import FeatureView
from asofline.demo.views import DEMO_REGISTRY
from asofline.online.store import FeatureVector, OnlineStore
from asofline.skew.logging import build_log_entry

if TYPE_CHECKING:
    from confluent_kafka import Producer


class OnlineFeaturesRequest(BaseModel):
    view: str
    entities: list[dict[str, str]] = Field(default_factory=list)
    as_of_ms: int | None = None


class OnlineFeaturesResponse(BaseModel):
    as_of_ms: int
    results: list[dict[str, Any]]


def _build_producer() -> Producer:
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.store = OnlineStore.from_url(SETTINGS.redis_url)
    app.state.producer = _build_producer()
    app.state.log_rng = Random(0)
    app.state.log_failures = 0
    # A background logging task holds no other reference once scheduled; asyncio keeps
    # only a weak reference to a task via ensure_future's return value, so an
    # unreferenced task can be garbage collected mid-flight. Every scheduled task is
    # added here and removed by its own done callback, which is what keeps it alive
    # until it actually finishes.
    app.state.background_log_tasks = set()
    try:
        yield
    finally:
        if app.state.background_log_tasks:
            await asyncio.gather(*app.state.background_log_tasks, return_exceptions=True)
        # flush(), not close(): confluent_kafka's Producer has no async close, and a
        # bounded flush gives buffered log entries a chance to actually reach the broker
        # before shutdown rather than dropping them silently.
        app.state.producer.flush(5.0)
        await app.state.store.close()


app = FastAPI(title="asofline online serving", lifespan=lifespan)


def _log_feature_vector(
    producer: Producer,
    view: FeatureView,
    entity_values: dict[str, str],
    *,
    request_ts_ms: int,
    served_at_ms: int,
    features: FeatureVector,
) -> None:
    """Build and publish one log entry. Any failure here is the caller's to swallow.

    Building the entry can itself raise (``FeatureLogError``) if the served vector's
    shape ever drifted from the view's declared features, which would otherwise be a
    silent, undetectable corruption of the detector's own input.
    """
    entry = build_log_entry(
        view,
        entity_values,
        log_id=uuid.uuid4().hex,
        request_ts_ms=request_ts_ms,
        served_at_ms=served_at_ms,
        features=features,
    )
    producer.produce(SETTINGS.feature_log_topic, value=json.dumps(entry.to_dict()).encode())
    producer.poll(0)  # Serve any pending delivery callbacks without blocking on I/O.


async def _log_in_background(
    app_state: Any,
    view: FeatureView,
    entity_values: dict[str, str],
    *,
    request_ts_ms: int,
    served_at_ms: int,
    features: FeatureVector,
) -> None:
    try:
        _log_feature_vector(
            app_state.producer,
            view,
            entity_values,
            request_ts_ms=request_ts_ms,
            served_at_ms=served_at_ms,
            features=features,
        )
    except Exception:
        app_state.log_failures += 1


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

    served_at_ms = int(time.time() * 1000)
    sample_rate = SETTINGS.feature_log_sample_rate
    for entity, features in zip(request.entities, vectors, strict=True):
        if sample_rate >= 1.0 or app.state.log_rng.random() < sample_rate:
            # Not awaited: the response below does not wait on this. See the module
            # docstring for why a logging failure must never become a request failure.
            # Tracked in background_log_tasks so it survives to completion; see the
            # comment on that set in lifespan() for why an untracked task is unsafe.
            task = asyncio.ensure_future(
                _log_in_background(
                    app.state,
                    view,
                    dict(entity),
                    request_ts_ms=as_of_ms,
                    served_at_ms=served_at_ms,
                    features=features,
                )
            )
            app.state.background_log_tasks.add(task)
            task.add_done_callback(app.state.background_log_tasks.discard)

    results = [
        {**entity, "features": features}
        for entity, features in zip(request.entities, vectors, strict=True)
    ]
    return OnlineFeaturesResponse(as_of_ms=as_of_ms, results=results)
