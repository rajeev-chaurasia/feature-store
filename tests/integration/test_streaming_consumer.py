"""Real-Redis tests for the Kafka-to-Redis consumer's event-processing core.

Every test here calls ``process_event`` directly, against the real, running Redis, with
no Kafka message and no mocking: the interesting behaviour (the tile read-modify-write,
the head push and trim, the TTLs, and the at-least-once double-count) all lives in how
``process_event`` touches Redis, and mocking that away would test nothing.

Redis is a shared instance a parallel agent's own test suite may be using at the same
moment, so this module targets ``tests.redis_fixtures.TEST_REDIS_URL`` (the database index
that suite set aside for tests, distinct from the default database an application would
use) rather than inventing a second convention, and cleans up only the keys it creates
itself, matched by a random marker embedded in every join-key value used here. Nothing
here calls ``FLUSHDB``: another suite's fixtures may be live in the same database at the
same time, and only the marker-matched keys are this test's to delete.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
import redis

from asofline.agg import brute_force, monoid_for
from asofline.agg.window import tile_index
from asofline.compiler.spec import tile_specs
from asofline.definitions.aggregation import AggFunction
from asofline.demo.events import EngagementEvent, EventType
from asofline.demo.views import DEMO_REGISTRY, USER_ENGAGEMENT, VIDEO_ENGAGEMENT
from asofline.online.codec import decode_state
from asofline.online.head import decode_head_event
from asofline.online.keys import entity_value_key, head_zset_key, tile_field, tile_hash_key
from asofline.streaming.consumer import process_event
from tests.redis_fixtures import TEST_REDIS_URL

pytestmark = pytest.mark.integration

FIVE_MIN_MS = 5 * 60 * 1000
ONE_HOUR_MS = 60 * 60 * 1000
BASE_TS = 1_700_000_000_000  # 2023-11-14T22:13:20Z, comfortably after the epoch


@pytest.fixture
def redis_client() -> Iterator[redis.Redis]:
    """A plain sync client, matching ``streaming.consumer``'s own (``confluent-kafka``'s
    poll loop is synchronous, so the consumer it feeds is written against ``redis.Redis``,
    not ``redis.asyncio``). Pointed at the same dedicated test database the parallel
    agent's async fixtures use, so this suite's data never lands in the default database.
    """
    client = redis.Redis.from_url(TEST_REDIS_URL)
    try:
        client.ping()
    except redis.exceptions.ConnectionError as error:
        pytest.skip(f"redis not reachable at {TEST_REDIS_URL}: {error}")
    yield client
    client.close()


@pytest.fixture
def marker() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture(autouse=True)
def _cleanup(redis_client: redis.Redis, marker: str) -> Iterator[None]:
    yield
    needle = marker.encode()
    stale = [key for key in redis_client.scan_iter(match="fs:*") if needle in key]
    if stale:
        redis_client.delete(*stale)


def _event(
    marker: str,
    *,
    event_type: EventType,
    event_ts: int,
    video: str = "v1",
    watch_seconds: float | None = None,
    liked: int = 0,
    shared: int = 0,
) -> EngagementEvent:
    return EngagementEvent(
        event_id=uuid.uuid4().hex,
        event_type=event_type,
        user_id=f"u-{marker}",
        video_id=f"{video}-{marker}",
        event_ts=event_ts,
        created_ts=event_ts,
        watch_seconds=watch_seconds,
        liked=liked,
        shared=shared,
    )


def _decode_field(
    client: redis.Redis, key: str, field: str, *, arity: int
) -> tuple[float, ...] | None:
    payload = client.hget(key, field)
    return None if payload is None else decode_state(payload, arity=arity)


def test_watch_events_fold_into_tiles_across_grids(redis_client: redis.Redis, marker: str) -> None:
    """Two watch events land in one 5-minute tile, a third in a different one, same hour.

    Exercises the two-grid fan-out the resolution ladder promises: ``watch_seconds_sum``
    is read by a 1h window (5-minute grid) and by 1d/7d windows (1-hour grid), so one event
    writes two tile fields, not one.
    """
    e1 = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS, watch_seconds=10.0)
    e2 = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS + 60_000, watch_seconds=20.0)
    e3 = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS + 400_000, watch_seconds=5.0)
    for event in (e1, e2, e3):
        process_event(redis_client, event, DEMO_REGISTRY)

    entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": e1.user_id})
    fine_key = tile_hash_key(USER_ENGAGEMENT, entity_key, FIVE_MIN_MS)
    coarse_key = tile_hash_key(USER_ENGAGEMENT, entity_key, ONE_HOUR_MS)

    fine_index_1 = tile_index(BASE_TS, FIVE_MIN_MS)
    fine_index_2 = tile_index(BASE_TS + 400_000, FIVE_MIN_MS)
    assert fine_index_1 != fine_index_2  # the two events really do land in different tiles

    sum_state_1 = _decode_field(
        redis_client, fine_key, tile_field("watch_seconds_sum", fine_index_1), arity=1
    )
    sum_state_2 = _decode_field(
        redis_client, fine_key, tile_field("watch_seconds_sum", fine_index_2), arity=1
    )
    assert sum_state_1 == (30.0,)  # 10 + 20, same 5-minute tile
    assert sum_state_2 == (5.0,)

    coarse_index = tile_index(BASE_TS, ONE_HOUR_MS)
    assert coarse_index == tile_index(BASE_TS + 400_000, ONE_HOUR_MS)  # same hour
    coarse_sum = _decode_field(
        redis_client, coarse_key, tile_field("watch_seconds_sum", coarse_index), arity=1
    )
    assert coarse_sum == (35.0,)  # all three fold into the one coarse tile

    coarse_count = _decode_field(
        redis_client, coarse_key, tile_field("count", coarse_index), arity=1
    )
    assert coarse_count == (3.0,)

    coarse_avg = _decode_field(
        redis_client, coarse_key, tile_field("watch_seconds_avg", coarse_index), arity=2
    )
    assert coarse_avg == (35.0, 3.0)

    # Cross-check against agg.rollup.brute_force, the project's own oracle: the finalized
    # 1-day sum at an as_of just after the last event must match the offline recomputation.
    as_of = BASE_TS + 400_000 + 1
    expected_sum = brute_force(
        monoid_for(AggFunction.SUM),
        [(e.event_ts, e.watch_seconds) for e in (e1, e2, e3)],
        as_of_ms=as_of,
        window_ms=24 * 60 * 60 * 1000,
        granularity_ms=ONE_HOUR_MS,
    )
    assert expected_sum == pytest.approx(35.0)


def test_impression_event_counts_but_does_not_touch_watch_seconds(
    redis_client: redis.Redis, marker: str
) -> None:
    event = _event(marker, event_type=EventType.IMPRESSION, event_ts=BASE_TS)
    process_event(redis_client, event, DEMO_REGISTRY)

    entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": event.user_id})
    fine_key = tile_hash_key(USER_ENGAGEMENT, entity_key, FIVE_MIN_MS)
    coarse_key = tile_hash_key(USER_ENGAGEMENT, entity_key, ONE_HOUR_MS)
    fine_index = tile_index(BASE_TS, FIVE_MIN_MS)
    coarse_index = tile_index(BASE_TS, ONE_HOUR_MS)

    count_state = _decode_field(redis_client, fine_key, tile_field("count", fine_index), arity=1)
    assert count_state == (1.0,)  # an impression is still an event

    sum_state = _decode_field(
        redis_client, fine_key, tile_field("watch_seconds_sum", fine_index), arity=1
    )
    assert sum_state is None  # never written: watch_seconds is None on an impression

    avg_state = _decode_field(
        redis_client, coarse_key, tile_field("watch_seconds_avg", coarse_index), arity=2
    )
    assert avg_state is None


def test_watch_event_fans_out_to_both_views(redis_client: redis.Redis, marker: str) -> None:
    event = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS, watch_seconds=42.0)
    process_event(redis_client, event, DEMO_REGISTRY)

    user_key = entity_value_key(USER_ENGAGEMENT, {"user_id": event.user_id})
    video_key = entity_value_key(VIDEO_ENGAGEMENT, {"video_id": event.video_id})
    fine_index = tile_index(BASE_TS, FIVE_MIN_MS)
    coarse_index = tile_index(BASE_TS, ONE_HOUR_MS)

    user_sum = _decode_field(
        redis_client,
        tile_hash_key(USER_ENGAGEMENT, user_key, FIVE_MIN_MS),
        tile_field("watch_seconds_sum", fine_index),
        arity=1,
    )
    video_sum = _decode_field(
        redis_client,
        tile_hash_key(VIDEO_ENGAGEMENT, video_key, FIVE_MIN_MS),
        tile_field("watch_seconds_sum", fine_index),
        arity=1,
    )
    assert user_sum == (42.0,)
    assert video_sum == (42.0,)

    video_max = _decode_field(
        redis_client,
        tile_hash_key(VIDEO_ENGAGEMENT, video_key, ONE_HOUR_MS),
        tile_field("watch_seconds_max", coarse_index),
        arity=1,
    )
    assert video_max == (42.0,)  # video_engagement's no-inverse aggregation, also updated


def test_redelivery_double_counts_sum_and_count(redis_client: redis.Redis, marker: str) -> None:
    """The documented at-least-once limitation, demonstrated rather than only asserted.

    Reprocessing the same event twice simulates a crash after the Redis write but before
    the offset commit, followed by redelivery. SUM and COUNT double-count in that case,
    which is the plan's accepted tradeoff for not building exactly-once dedup, not a bug
    this consumer is meant to hide.
    """
    event = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS, watch_seconds=7.0)
    process_event(redis_client, event, DEMO_REGISTRY)
    process_event(redis_client, event, DEMO_REGISTRY)  # redelivery of the exact same event

    entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": event.user_id})
    fine_key = tile_hash_key(USER_ENGAGEMENT, entity_key, FIVE_MIN_MS)
    fine_index = tile_index(BASE_TS, FIVE_MIN_MS)

    sum_state = _decode_field(
        redis_client, fine_key, tile_field("watch_seconds_sum", fine_index), arity=1
    )
    count_state = _decode_field(redis_client, fine_key, tile_field("count", fine_index), arity=1)

    assert sum_state == (14.0,)  # 7.0 applied twice: double-counted, as documented
    assert count_state == (2.0,)  # the same event counted as two events


def test_head_zset_trims_entries_outside_any_grids_reach(
    redis_client: redis.Redis, marker: str
) -> None:
    entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": f"u-{marker}"})
    key = head_zset_key(USER_ENGAGEMENT, entity_key)

    # Seed an old entry, further back than any grid's head window could ever reach once a
    # new event's trim runs (the coarsest grid user_engagement writes to is the 1-hour
    # grid, so anything older than one hour before the new event cannot be needed again).
    old_ts = BASE_TS - 3 * ONE_HOUR_MS
    old_member = '{"planted": true}'
    redis_client.zadd(key, {old_member: old_ts})

    event = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS, watch_seconds=1.0)
    process_event(redis_client, event, DEMO_REGISTRY)

    assert redis_client.zscore(key, old_member) is None  # trimmed
    members = redis_client.zrange(key, 0, -1)
    assert len(members) == 1  # only the new event's member survives
    decoded = decode_head_event(members[0].decode())
    assert decoded["watch_seconds"] == 1.0


def test_ttls_are_set_on_every_written_key(redis_client: redis.Redis, marker: str) -> None:
    event = _event(marker, event_type=EventType.WATCH, event_ts=BASE_TS, watch_seconds=3.0)
    process_event(redis_client, event, DEMO_REGISTRY)

    entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": event.user_id})
    fine_key = tile_hash_key(USER_ENGAGEMENT, entity_key, FIVE_MIN_MS)
    coarse_key = tile_hash_key(USER_ENGAGEMENT, entity_key, ONE_HOUR_MS)
    head_key = head_zset_key(USER_ENGAGEMENT, entity_key)

    for key in (fine_key, coarse_key, head_key):
        ttl_ms = redis_client.pttl(key)
        assert ttl_ms > 0, f"{key} has no TTL set"

    # Each grid's hash key is shared by several aggregations with potentially different
    # retentions; the key's TTL must be at least the longest of them; it is refreshed to
    # exactly that maximum on every write (see consumer._grid_retention_ms), so it should
    # never exceed it either, modulo the trivial clock skew of the test itself running.
    specs = tile_specs(USER_ENGAGEMENT)
    fine_retention = max(s.retention_ms for s in specs if s.granularity_ms == FIVE_MIN_MS)
    coarse_retention = max(s.retention_ms for s in specs if s.granularity_ms == ONE_HOUR_MS)
    assert redis_client.pttl(fine_key) <= fine_retention
    assert redis_client.pttl(coarse_key) <= coarse_retention
