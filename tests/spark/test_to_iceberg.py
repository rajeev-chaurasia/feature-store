"""P4 done-test, raw-landing half: Kafka events actually land in the Iceberg raw table.

Events are produced with a ``confluent-kafka`` ``Producer`` rather than through Spark's
own Kafka writer. ``confluent-kafka`` is already a project dependency (the ``online``
extra, for the future Kafka-to-Redis consumer this job's plan pairs with), and driving it
directly gives per-message control over the exact JSON wire format
(``EngagementEvent.to_dict()``) with no second Spark write path needed just to seed a
topic.

``engagement_events`` is the real, shared topic (``SETTINGS.events_topic``), the same one
the plan says a second, independent consumer will read. Rather than using a private test
topic, every event this module produces carries a run-unique prefix in ``event_id``, and
every assertion filters on that prefix. That makes the test correct even if another
process is producing to the same topic at the same time, and it is the reason
``startingOffsets="earliest"`` is safe to use: a bounded run drains the whole topic, but
only rows carrying this run's prefix are ever asserted on.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from confluent_kafka import Producer
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_millis

from asofline.config import SETTINGS
from asofline.demo.events import EngagementEvent, EventType
from asofline.streaming.to_iceberg import run

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_stream_test"
TABLE = "engagement_events"

# Every test here uses the shared, session-scoped `spark` fixture from conftest.py rather
# than a private session of its own. SparkSession.builder.getOrCreate() returns a JVM-wide
# singleton, so a fixture believing it owns an isolated session and calling .stop() on it
# at module teardown actually stops the *shared* session for every later test in the
# process. See offline.session.KAFKA_PACKAGE's docstring for the other half of this fix:
# the shared session now always carries the Kafka connector jar this file's tests need.


def _produce(events: list[EngagementEvent]) -> None:
    producer = Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})
    for event in events:
        producer.produce(SETTINGS.events_topic, json.dumps(event.to_dict()).encode("utf-8"))
    producer.flush(timeout=30)


def _make_events(prefix: str) -> list[EngagementEvent]:
    base = 1_800_000_000_000  # fixed epoch-ms anchor, arbitrary but deterministic
    return [
        EngagementEvent(
            event_id=f"{prefix}-0",
            event_type=EventType.WATCH,
            user_id="u1",
            video_id="v1",
            event_ts=base,
            created_ts=base + 500,
            watch_seconds=12.5,
            liked=0,
            shared=0,
        ),
        EngagementEvent(
            event_id=f"{prefix}-1",
            event_type=EventType.IMPRESSION,
            user_id="u1",
            video_id="v2",
            event_ts=base + 1_000,
            created_ts=base + 1_000,
            watch_seconds=None,
            liked=0,
            shared=0,
        ),
        EngagementEvent(
            event_id=f"{prefix}-2",
            event_type=EventType.LIKE,
            user_id="u2",
            video_id="v1",
            event_ts=base + 2_000,
            created_ts=base + 2_400,
            watch_seconds=None,
            liked=1,
            shared=0,
        ),
        EngagementEvent(
            event_id=f"{prefix}-3",
            event_type=EventType.SHARE,
            user_id="u2",
            video_id="v3",
            event_ts=base + 3_000,
            created_ts=base + 3_000,
            watch_seconds=None,
            liked=0,
            shared=1,
        ),
    ]


def _rows_for_prefix(spark: SparkSession, prefix: str) -> list:
    table = f"{NAMESPACE}.{TABLE}"
    return (
        spark.sql(f"SELECT * FROM {table} WHERE event_id LIKE '{prefix}-%'")
        .withColumn("event_ts_ms", unix_millis("event_ts"))
        .withColumn("created_ts_ms", unix_millis("created_ts"))
        .orderBy("event_id")
        .collect()
    )


def test_events_land_with_correct_timestamps_and_fields(
    spark: SparkSession, tmp_path: Path
) -> None:
    """The core done-test: produce, run bounded, read back, check shape and values.

    Timestamps are compared with ``unix_millis()`` computed inside Spark SQL, not with a
    driver-side ``datetime`` collected out of the TIMESTAMP column. That collect path
    silently wears the driver's local time zone
    (``test_timestamp_columns_render_in_the_driver_timezone`` in test_smoke_iceberg.py
    documents exactly this trap), and comparing millis-in, millis-out entirely avoids it.
    """
    prefix = f"land-{uuid.uuid4().hex[:8]}"
    events = _make_events(prefix)
    _produce(events)

    query = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=str(tmp_path / "checkpoint"),
        bounded=True,
    )
    assert not query.isActive

    rows = _rows_for_prefix(spark, prefix)
    assert len(rows) == len(events)

    by_id = {row["event_id"]: row for row in rows}
    for event in events:
        row = by_id[event.event_id]
        assert row["event_ts_ms"] == event.event_ts
        assert row["created_ts_ms"] == event.created_ts
        assert row["event_type"] == str(event.event_type)
        assert row["liked"] == event.liked
        assert row["shared"] == event.shared
        if event.watch_seconds is None:
            assert row["watch_seconds"] is None
        else:
            assert row["watch_seconds"] == pytest.approx(event.watch_seconds)


def test_restart_against_the_same_checkpoint_does_not_duplicate(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Structured Streaming's checkpoint, not application-level dedup, is what makes a
    second bounded run against the same checkpoint a no-op when no new data arrived."""
    prefix = f"restart-{uuid.uuid4().hex[:8]}"
    events = _make_events(prefix)
    _produce(events)

    checkpoint = str(tmp_path / "checkpoint")
    first = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=checkpoint,
        bounded=True,
    )
    assert not first.isActive
    first_rows = _rows_for_prefix(spark, prefix)
    assert len(first_rows) == len(events)

    second = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=checkpoint,
        bounded=True,
    )
    assert not second.isActive
    second_rows = _rows_for_prefix(spark, prefix)
    assert len(second_rows) == len(events), "restart against the same checkpoint duplicated rows"


def test_uses_its_own_consumer_group(spark: SparkSession, tmp_path: Path) -> None:
    """Guards the plan's explicit requirement: this job must not share a consumer group
    with the Kafka-to-Redis consumer built separately against the same topic."""
    from asofline.streaming.to_iceberg import CONSUMER_GROUP

    assert CONSUMER_GROUP != "asofline-to-redis"
    assert "iceberg" in CONSUMER_GROUP

    prefix = f"group-{uuid.uuid4().hex[:8]}"
    _produce(_make_events(prefix)[:1])
    query = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=str(tmp_path / "checkpoint"),
        group_id=CONSUMER_GROUP,
        bounded=True,
    )
    assert not query.isActive
    assert len(_rows_for_prefix(spark, prefix)) == 1
