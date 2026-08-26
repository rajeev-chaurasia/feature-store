"""P5 done-test's input pipe: served feature vectors actually land, exploded, in Iceberg.

This job is not the skew detector; it only lands the detector's own evidence. The
assertions here are about the pipe, not about skew: the right number of exploded rows
appear, entity keys round-trip as a map lookup, timestamps round-trip in millis, and a
null feature value survives as SQL ``NULL`` rather than some sentinel.

Messages are produced with a ``confluent-kafka`` ``Producer`` directly onto the real,
shared ``feature_logs`` topic, the same one the serving app's own test suite publishes to.
Every entry here carries a run-unique ``user_id`` (a ``uuid4`` fragment), and every
assertion filters on it, so this test is correct regardless of what else is already on the
topic.
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
from asofline.skew.logging import FeatureLogEntry
from asofline.streaming.feature_log_to_iceberg import run

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_feature_log_test"
TABLE = "serving_log"

# Every test here uses the shared, session-scoped `spark` fixture from conftest.py rather
# than a private session of its own. SparkSession.builder.getOrCreate() returns a JVM-wide
# singleton, so a fixture believing it owns an isolated session and calling .stop() on it
# at module teardown actually stops the *shared* session for every later test in the
# process. See offline.session.KAFKA_PACKAGE's docstring for the other half of this fix:
# the shared session now always carries the Kafka connector jar this file's tests need.


def _produce(entries: list[FeatureLogEntry]) -> None:
    producer = Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})
    for entry in entries:
        producer.produce(SETTINGS.feature_log_topic, json.dumps(entry.to_dict()).encode("utf-8"))
    producer.flush(timeout=30)


def _make_entries(user_id: str) -> list[FeatureLogEntry]:
    """Three entries built to exercise the three things this test must prove:

    - ``multi`` has more than one feature, proving one input message explodes into
      several output rows rather than a 1:1 passthrough.
    - ``nully`` has a null feature value, the legitimate shape for a stale/unseen entity.
    - ``single`` is the plain case, included so the row-count arithmetic below is not
      accidentally true only because every entry happens to look alike.
    """
    base = 1_800_000_000_000  # fixed epoch-ms anchor, arbitrary but deterministic
    return [
        FeatureLogEntry(
            log_id=f"{user_id}-multi",
            view_name="user_engagement",
            view_version=1,
            entity_keys={"user_id": user_id},
            request_ts_ms=base,
            served_at_ms=base + 5,
            features={"watch_seconds_sum_1h": 12.5, "like_count_1h": 3.0, "share_count_1h": 0.0},
        ),
        FeatureLogEntry(
            log_id=f"{user_id}-nully",
            view_name="user_engagement",
            view_version=1,
            entity_keys={"user_id": user_id},
            request_ts_ms=base + 1_000,
            served_at_ms=base + 1_010,
            features={"watch_seconds_sum_1h": None},
        ),
        FeatureLogEntry(
            log_id=f"{user_id}-single",
            view_name="user_engagement",
            view_version=1,
            entity_keys={"user_id": user_id},
            request_ts_ms=base + 2_000,
            served_at_ms=base + 2_020,
            features={"like_count_1h": 7.0},
        ),
    ]


def _rows_for_user(spark: SparkSession, user_id: str) -> list:
    table = f"{NAMESPACE}.{TABLE}"
    return (
        spark.sql(f"SELECT * FROM {table} WHERE entity_keys['user_id'] = '{user_id}'")
        .withColumn("request_ts_ms", unix_millis("request_ts"))
        .withColumn("served_at_ts_ms", unix_millis("served_at_ts"))
        .orderBy("log_id", "feature_name")
        .collect()
    )


def test_feature_logs_explode_into_the_serving_log_table(
    spark: SparkSession, tmp_path: Path
) -> None:
    user_id = f"u_flti_{uuid.uuid4().hex[:12]}"
    entries = _make_entries(user_id)
    _produce(entries)

    query = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=str(tmp_path / "checkpoint"),
        bounded=True,
    )
    assert not query.isActive

    rows = _rows_for_user(spark, user_id)

    # entries * features_per_entry: 3 + 1 + 1 = 5 exploded rows from 3 messages.
    expected_row_count = sum(len(entry.features) for entry in entries)
    assert len(rows) == expected_row_count

    by_log_and_feature = {(row["log_id"], row["feature_name"]): row for row in rows}

    # Multi-feature entry: proves one message explodes into multiple rows, and each row
    # carries the entity keys, view metadata, and timestamps of its parent message.
    multi_entry = entries[0]
    for feature_name, value in multi_entry.features.items():
        row = by_log_and_feature[(multi_entry.log_id, feature_name)]
        assert row["view_name"] == multi_entry.view_name
        assert row["view_version"] == multi_entry.view_version
        assert row["entity_keys"]["user_id"] == user_id
        assert row["request_ts_ms"] == multi_entry.request_ts_ms
        assert row["served_at_ts_ms"] == multi_entry.served_at_ms
        assert row["value"] == pytest.approx(value)

    # Null feature value: must land as SQL NULL, not a sentinel like 0.0 or NaN.
    null_entry = entries[1]
    null_row = by_log_and_feature[(null_entry.log_id, "watch_seconds_sum_1h")]
    assert null_row["value"] is None

    # Single-feature entry: the plain case, one input row producing exactly one output row.
    single_entry = entries[2]
    single_row = by_log_and_feature[(single_entry.log_id, "like_count_1h")]
    assert single_row["value"] == pytest.approx(7.0)


def test_uses_its_own_consumer_group(spark: SparkSession, tmp_path: Path) -> None:
    """Guards the plan's requirement: three independent consumer groups on two topics, so
    a stall in this job is never mistaken for a stall in either the raw-events-to-Iceberg
    job or the Kafka-to-Redis tile consumer."""
    from asofline.streaming.feature_log_to_iceberg import CONSUMER_GROUP

    assert CONSUMER_GROUP not in {"asofline-to-iceberg", "asofline-to-redis"}
    assert "feature-log" in CONSUMER_GROUP

    user_id = f"u_flti_group_{uuid.uuid4().hex[:12]}"
    _produce(_make_entries(user_id)[:1])
    query = run(
        spark,
        namespace=NAMESPACE,
        table=TABLE,
        checkpoint_location=str(tmp_path / "checkpoint"),
        group_id=CONSUMER_GROUP,
        bounded=True,
    )
    assert not query.isActive
    rows = _rows_for_user(spark, user_id)
    assert len(rows) == 3  # the first entry's three features
