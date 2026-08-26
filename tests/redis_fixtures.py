"""Redis seeding helpers for tests that exercise the online store.

Rather than requiring a running Kafka-to-Redis consumer, tests here write directly
through the same primitives it uses: ``HSET`` for tile fields via
``asofline.online.codec.encode_state``, and ``ZADD`` for head events via
``asofline.online.head.encode_head_event``. This lets the online store's own tests stay
independent of the consumer's, and vice versa, while both still exercise real Redis.

Tests run against ``TEST_REDIS_URL``, a database index set aside for this suite, never the
default database (index 0) an application, a concurrently running consumer, or another
agent's own test run might be using. ``flush_test_database`` only ever issues ``FLUSHDB``
after that database has been selected, never ``FLUSHALL``.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Mapping

from redis.asyncio import Redis

from asofline.agg.window import tile_index
from asofline.definitions.view import FeatureView
from asofline.online.codec import encode_state
from asofline.online.head import encode_head_event
from asofline.online.keys import entity_value_key, head_zset_key, tile_field, tile_hash_key

TEST_REDIS_URL = "redis://localhost:6379/15"
"""An index dedicated to this test suite, so seeded fixtures cannot collide with a
concurrently running consumer, a real deployment, or another agent's own test data, all of
which are expected to stay on database 0."""


def run_async[**P, T](fn: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    """Run an ``async def`` test body to completion under plain pytest.

    Neither ``pytest-asyncio`` nor the ``anyio`` pytest plugin is a declared dependency
    (see ``pyproject.toml``), so this is the smallest way to write async test bodies
    against a real Redis connection without adding one.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def make_test_redis() -> Redis:
    return Redis.from_url(TEST_REDIS_URL)


async def flush_test_database(redis: Redis) -> None:
    """Clear the dedicated test database. Never touches any other database."""
    await redis.flushdb()


async def seed_tile(
    redis: Redis,
    view: FeatureView,
    values: Mapping[str, str],
    *,
    agg_name: str,
    granularity_ms: int,
    event_ts_ms: int,
    state: tuple[float, ...],
) -> None:
    """Write one tile field in the exact shape ``online.store`` expects to ``HGETALL``.

    ``event_ts_ms`` rather than a raw tile index, so a test can say "this state covers the
    event at this moment" without doing the index arithmetic itself.
    """
    entity_key = entity_value_key(view, dict(values))
    key = tile_hash_key(view, entity_key, granularity_ms)
    field = tile_field(agg_name, tile_index(event_ts_ms, granularity_ms))
    await redis.hset(key, field, encode_state(state))


async def seed_head_event(
    redis: Redis,
    view: FeatureView,
    values: Mapping[str, str],
    *,
    event_ts_ms: int,
    columns: Mapping[str, float | None],
    event_id: str | None = None,
) -> None:
    """Push one raw event onto the head ZSET in the exact shape ``online.store`` reads.

    ``columns`` maps a source column name (``FeatureSpec.column``, e.g.
    ``"watch_seconds"``) to its value on this event, or ``None`` if the event does not
    carry that column at all.

    ``event_id`` defaults to a value derived from ``event_ts_ms`` for callers seeding a
    single head event per entity, which is every existing caller. It must be given
    explicitly, and be distinct, when seeding more than one head event with the same
    ``columns`` for the same entity: the real encoding keys each ZSET member on
    ``event_id`` specifically so that two events with identical column values (every
    impression in this project's demo data, for instance) do not collide into one entry,
    and a fixed default here would silently reintroduce that same collision in tests that
    seed more than one such event.
    """
    entity_key = entity_value_key(view, dict(values))
    key = head_zset_key(view, entity_key)
    resolved_event_id = event_id if event_id is not None else f"seed-{event_ts_ms}"
    member = encode_head_event(resolved_event_id, columns)
    await redis.zadd(key, {member: float(event_ts_ms)})
