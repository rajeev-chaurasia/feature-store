"""Kafka to Redis: the freshness half of P4.

This is one of two independent consumers of ``engagement_events``. The other,
``streaming.to_iceberg``, lands raw events in Iceberg on its own consumer group. The plan
is explicit that the two must not share a group: two consumers, one topic, independent
lag, so a stall in one is never mistaken for a stall in the other.

**Manual offset commit, after the Redis write, not before and not via Kafka's
auto-commit.** A crash between the Redis write and the commit redelivers the batch, and
the redelivered events get folded into Redis state a second time. For ``SUM`` and
``COUNT`` that is a real double-count; ``MIN``/``MAX`` are idempotent under
re-application, and re-adding the same value to an ``AVG`` state double-counts exactly
like ``SUM``. This is the plan's accepted at-least-once limitation, not a bug: building
exactly-once dedup (a per-partition, per-key seen-offsets window) is explicitly out of
scope, and ``tests/integration/test_streaming_consumer.py`` demonstrates the double-count
directly rather than only asserting it in prose.

**Atomicity of one tile field's read-modify-write.** A hash field's new state depends on
its old state (``Monoid.merge``), and two writers touching the same field at once would
lose an update if the read-then-write were unguarded. The merge itself has to stay in
Python: it is ``asofline.agg.monoid``, the one implementation both the batch and the
streaming path are built to share, and reimplementing float decode/merge/encode in a Lua
script would fork that contract into a second, Lua-flavoured copy of ``online.codec`` and
``agg.monoid`` with its own chance to drift. Instead, ``_merge_field`` uses the redis-py
client's optimistic ``WATCH``/``MULTI`` transaction: it watches the hash key, reads the
field, computes the new state in Python, and commits only if nothing touched the key in
between, retrying on a ``WatchError``. That is real atomicity (no lost update, from any
number of concurrent writers touching the same key), it costs nothing extra in the
uncontended case this single-instance demo actually runs, and it keeps the merge algebra
in exactly one place.
"""

from __future__ import annotations

import json
import logging

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition
from redis import Redis
from redis.exceptions import WatchError

from asofline.agg.monoid import Monoid, monoid_for
from asofline.agg.window import tile_index
from asofline.compiler.spec import TileSpec, tile_specs
from asofline.config import SETTINGS, Settings
from asofline.definitions.aggregation import AggFunction
from asofline.definitions.registry import Registry
from asofline.definitions.view import FeatureView
from asofline.demo.events import EngagementEvent
from asofline.online.codec import decode_state, encode_state
from asofline.online.head import encode_head_event
from asofline.online.keys import entity_value_key, head_zset_key, tile_field, tile_hash_key

logger = logging.getLogger(__name__)

# Distinct from streaming.to_iceberg.CONSUMER_GROUP, per the plan's "own consumer group"
# requirement: this consumer's lag and that one's must be observable independently.
CONSUMER_GROUP = "asofline-to-redis"

# All join keys this project's events can supply. A view's entities are looked up against
# this rather than something narrower so adding a view keyed on a column already on
# EngagementEvent needs no change here.
_EVENT_JOIN_VALUES = ("user_id", "video_id")


def build_consumer(*, settings: Settings = SETTINGS, group_id: str = CONSUMER_GROUP) -> Consumer:
    """A consumer with auto-commit off. Offsets move only after ``run`` commits them."""
    return Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )


def build_redis_client(*, settings: Settings = SETTINGS) -> Redis:
    return Redis.from_url(settings.redis_url)


def _join_values(event: EngagementEvent) -> dict[str, str]:
    return {key: getattr(event, key) for key in _EVENT_JOIN_VALUES}


def _lift_value(spec: TileSpec, event: EngagementEvent) -> float | None:
    """What one event contributes to one tile spec, or ``None`` if it contributes nothing.

    Mirrors ``offline.tiles.state_expressions``' null handling for the single-event case:
    ``COUNT`` counts the row regardless of its columns (the lifted value is thrown away by
    ``Monoid.lift`` for ``COUNT``, so any float will do). Every other function reads
    ``spec.column`` off the event, and a ``None`` there (``watch_seconds`` on a non-watch
    event) means this event does not contribute to this aggregation at all, not that it
    contributes zero.
    """
    if spec.function is AggFunction.COUNT:
        return 1.0
    assert spec.column is not None  # every non-COUNT function declares a column
    raw = getattr(event, spec.column)
    return None if raw is None else float(raw)


def _contributions(specs: tuple[TileSpec, ...], event: EngagementEvent) -> dict[TileSpec, float]:
    contributions: dict[TileSpec, float] = {}
    for spec in specs:
        value = _lift_value(spec, event)
        if value is not None:
            contributions[spec] = value
    return contributions


def _merge_field(
    redis_client: Redis,
    key: str,
    field: str,
    monoid: Monoid,
    lifted: tuple[float, ...],
    *,
    ttl_ms: int,
) -> None:
    """Atomically fold ``lifted`` into whatever state currently sits at ``key``/``field``.

    See the module docstring for why this is a WATCH/MULTI transaction rather than a Lua
    script or an unguarded read-then-write.
    """
    with redis_client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)  # type: ignore[no-untyped-call]
                current = pipe.hget(key, field)
                if current is None:
                    state = monoid.identity
                else:
                    # decode_responses is never set on this client, so this is always
                    # bytes; the broader `bytes | str | None` in redis-py's stub is for
                    # clients that might have decode_responses=True.
                    assert isinstance(current, bytes)
                    state = decode_state(current, arity=monoid.arity)
                new_state = monoid.merge(state, lifted)
                pipe.multi()
                pipe.hset(key, field, encode_state(new_state))
                pipe.pexpire(key, ttl_ms)
                pipe.execute()
                return
            except WatchError:
                # Another writer touched this key between WATCH and EXEC. Retry: the
                # re-read picks up whatever it left behind, so nothing is lost.
                continue


def _head_columns(specs: tuple[TileSpec, ...]) -> tuple[str, ...]:
    """The distinct, sorted, non-``None`` column names this view's tile specs reference.

    Sorting is not part of the wire contract (``online.head.encode_head_event`` sorts its
    own JSON keys), but the columns are read straight off the event by name here, so a
    deterministic iteration order keeps this function's own behaviour reproducible.
    """
    return tuple(sorted({spec.column for spec in specs if spec.column is not None}))


def _push_head_event(
    redis_client: Redis,
    view: FeatureView,
    entity_key: str,
    event: EngagementEvent,
    specs: tuple[TileSpec, ...],
) -> None:
    """Append the raw event to the head ZSET and trim what it can never need again.

    The trim boundary is the coarsest grid this view writes to: for any grid ``g``, the
    widest a head interval ``[align_down(T, g), T)`` can ever be is ``g`` itself
    (``agg.window.bounds_for``), so an entry older than ``event_ts - max(grid widths)``
    cannot be inside the head window of *any* grid this view has, no matter what ``T`` a
    later read asks for. Using this event's own timestamp as the trim reference (rather
    than a wall-clock "now") is safe under out-of-order delivery: an event that arrives
    late trims to an earlier boundary than one already applied would have, which only
    keeps more than strictly necessary, never less.

    The TTL matches ``view.ttl``, the same online staleness bound the tile hashes are
    read under, so an idle head ZSET ages out on the same schedule as the values it would
    otherwise support rather than on the (much shorter) coarsest grid width, which would
    let the head disappear while the tiles it complements are still considered fresh.
    """
    columns = _head_columns(specs)
    mapping = {column: getattr(event, column) for column in columns}
    member = encode_head_event(event.event_id, mapping)
    key = head_zset_key(view, entity_key)
    coarsest_grid_ms = max(spec.granularity_ms for spec in specs)
    cutoff_ms = event.event_ts - coarsest_grid_ms
    ttl_ms = int(view.ttl.total_seconds() * 1000)
    with redis_client.pipeline(transaction=True) as pipe:
        pipe.zadd(key, {member: event.event_ts})
        pipe.zremrangebyscore(key, "-inf", cutoff_ms)
        pipe.pexpire(key, ttl_ms)
        pipe.execute()


def process_event(redis_client: Redis, event: EngagementEvent, registry: Registry) -> None:
    """Fold one event into every view it feeds.

    An event fans out to every view whose join key it carries *and* which has at least
    one tile spec it actually contributes to (``_contributions`` is non-empty for that
    view). In this registry every event carries both ``user_id`` and ``video_id``, so a
    single ``watch`` event updates both ``user_engagement`` and ``video_engagement``; a
    view is skipped only when the event has nothing at all for it (never happens in this
    registry, since every view has either a ``COUNT`` or an always-populated ``liked``/
    ``shared`` column, but the check keeps this correct for a registry that adds a view
    without such a column).
    """
    join_values = _join_values(event)
    for view in registry.views:
        try:
            entity_key = entity_value_key(view, join_values)
        except KeyError:
            continue
        specs = tile_specs(view)
        contributions = _contributions(specs, event)
        if not contributions:
            continue
        grid_ttl_ms = _grid_retention_ms(specs)
        for spec, value in contributions.items():
            monoid = monoid_for(spec.function)
            key = tile_hash_key(view, entity_key, spec.granularity_ms)
            field = tile_field(spec.agg_name, tile_index(event.event_ts, spec.granularity_ms))
            _merge_field(
                redis_client,
                key,
                field,
                monoid,
                monoid.lift(value),
                ttl_ms=grid_ttl_ms[spec.granularity_ms],
            )
        _push_head_event(redis_client, view, entity_key, event, specs)


def _grid_retention_ms(specs: tuple[TileSpec, ...]) -> dict[int, int]:
    """The TTL to apply to each grid's shared hash key.

    Several aggregations on the same grid share one Redis hash (``online.keys``'s whole
    point: one ``HGETALL`` per grid, not one per aggregation), and ``PEXPIRE`` sets a
    single TTL for the entire key. Using a just-written field's own ``retention_ms`` would
    let a short-retention field (say ``count`` on a 1-day window) shrink the key's TTL
    below what a long-retention sibling field on the same key still needs (say a 7-day
    ``sum``), evicting live data early. The key must outlive the longest retention among
    everything it holds, so every write refreshes the TTL to the grid's maximum rather
    than to the one field's own.
    """
    retention: dict[int, int] = {}
    for spec in specs:
        retention[spec.granularity_ms] = max(
            retention.get(spec.granularity_ms, 0), spec.retention_ms
        )
    return retention


def _offsets_to_commit(messages: list[Message]) -> list[TopicPartition]:
    """The highest offset seen per partition in this batch, committed as offset + 1.

    ``Consumer.commit(message=...)`` only covers one partition; a batch drained from a
    6-partition topic needs one entry per partition actually present in it.
    """
    highest: dict[tuple[str, int], int] = {}
    for message in messages:
        topic, partition, offset = message.topic(), message.partition(), message.offset()
        # A message pulled off a live consumer always carries all three; only a message
        # built by hand (never done here) could leave one unset.
        assert topic is not None
        assert partition is not None
        assert offset is not None
        key = (topic, partition)
        highest[key] = max(highest.get(key, -1), offset)
    return [
        TopicPartition(topic, partition, offset + 1)
        for (topic, partition), offset in highest.items()
    ]


def _poll_batch(consumer: Consumer, poll_timeout: float, max_batch: int) -> list[Message]:
    messages: list[Message] = []
    while len(messages) < max_batch:
        # Block for poll_timeout only on the first poll of a batch; once something has
        # arrived, drain whatever else is immediately available rather than waiting again.
        message = consumer.poll(poll_timeout if not messages else 0.0)
        if message is None:
            break
        error = message.error()
        if error is not None:
            if error.code() == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(error)
        messages.append(message)
    return messages


def run(
    consumer: Consumer,
    redis_client: Redis,
    registry: Registry,
    *,
    topic: str = SETTINGS.events_topic,
    poll_timeout: float = 1.0,
    max_batch: int = 500,
    stop_after_empty_polls: int | None = None,
) -> None:
    """Drain ``topic`` into ``redis_client`` forever, or until ``stop_after_empty_polls``.

    The loop itself is deliberately thin: parsing and Redis writes live in
    ``process_event``, which needs no Kafka message and no running consumer to test. This
    function's only job is pulling a batch, applying it, and then committing, in that
    order, so the manual-commit-after-write guarantee lives in one obvious place.
    """
    consumer.subscribe([topic])
    empty_polls = 0
    try:
        while True:
            messages = _poll_batch(consumer, poll_timeout, max_batch)
            if not messages:
                empty_polls += 1
                if stop_after_empty_polls is not None and empty_polls >= stop_after_empty_polls:
                    return
                continue
            empty_polls = 0
            for message in messages:
                payload = message.value()
                assert payload is not None
                event = EngagementEvent.from_dict(json.loads(payload))
                process_event(redis_client, event, registry)
            consumer.commit(offsets=_offsets_to_commit(messages), asynchronous=False)
    finally:
        consumer.close()


if __name__ == "__main__":  # pragma: no cover
    from asofline.demo.views import DEMO_REGISTRY

    logging.basicConfig(level=logging.INFO)
    _consumer = build_consumer()
    _redis = build_redis_client()
    try:
        run(_consumer, _redis, DEMO_REGISTRY)
    finally:
        _redis.close()
