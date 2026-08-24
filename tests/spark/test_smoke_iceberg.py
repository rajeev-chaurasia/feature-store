"""P0 done-test: the stack is real.

This is the gate that decides whether the Spark 4.0 plus Iceberg 1.11 lane holds or
whether the project falls back to Spark 3.5. It is deliberately end to end rather than
mocked: it proves the REST catalog answers, that S3FileIO can write to MinIO, and that
what came back is what went in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from asofline.demo.events import from_millis, to_millis
from asofline.offline.session import (
    SUPPORTED_JAVA_MAJORS,
    JavaVersionError,
    assert_supported_java,
    catalog_options,
)

pytestmark = pytest.mark.spark


def test_java_gate_accepts_the_configured_jdk() -> None:
    assert assert_supported_java() in SUPPORTED_JAVA_MAJORS


def test_java_gate_rejects_a_missing_jdk() -> None:
    from dataclasses import replace

    from asofline.config import Settings

    broken = replace(Settings.from_env(), java_home="/nonexistent/jdk")
    with pytest.raises(JavaVersionError, match="no java binary"):
        assert_supported_java(broken)


def test_catalog_is_wired_to_rest_and_s3_file_io() -> None:
    options = catalog_options()
    assert options["spark.sql.catalog.asofline.type"] == "rest"
    assert options["spark.sql.catalog.asofline.io-impl"].endswith("S3FileIO")


def test_round_trip_through_the_rest_catalog(spark: SparkSession, test_namespace: str) -> None:
    """Create, write, read back, and prove it went through Iceberg.

    The table carries the same instant twice, once as a ``TIMESTAMP`` and once as the
    ``BIGINT`` epoch-millisecond encoding this project uses everywhere below the boundary
    layers. Both are asserted, because the difference between them is the reason for that
    decision and is documented in ``test_timestamp_columns_render_in_the_driver_timezone``.
    """
    table = f"{test_namespace}.smoke"
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql(
        f"""
        CREATE TABLE {table} (
            entity_id STRING NOT NULL,
            event_ts  TIMESTAMP,
            event_ms  BIGINT,
            value     DOUBLE
        ) USING iceberg
        PARTITIONED BY (days(event_ts))
        """
    )
    moment = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)
    millis = to_millis(moment)
    spark.createDataFrame(
        [("u1", moment, millis, 1.5)],
        "entity_id string, event_ts timestamp, event_ms bigint, value double",
    ).writeTo(table).append()

    rows = spark.sql(f"SELECT entity_id, event_ts, event_ms, value FROM {table}").collect()
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "u1"
    assert rows[0]["value"] == pytest.approx(1.5)

    # The epoch encoding survives as an integer, with no conversion anywhere to get wrong.
    assert rows[0]["event_ms"] == millis
    assert from_millis(rows[0]["event_ms"]) == moment

    # The instant in the TIMESTAMP column also survived. Only its rendering moved.
    assert rows[0]["event_ts"].astimezone(UTC) == moment

    # The snapshot metadata table is the cheapest proof that this went through Iceberg
    # rather than falling back to some other provider.
    snapshots = spark.sql(f"SELECT * FROM {table}.snapshots").collect()
    assert len(snapshots) == 1

    spark.sql(f"DROP TABLE {table} PURGE")


def test_timestamp_columns_render_in_the_driver_timezone(
    spark: SparkSession, test_namespace: str
) -> None:
    """Pin down the trap, so a future reader does not rediscover it the hard way.

    ``build_session`` sets ``spark.sql.session.timeZone=UTC``. That governs SQL-side casts,
    but PySpark's ``collect()`` turns the internal epoch value into a **naive** datetime
    using the driver process's local zone, which this machine reports as PDT. So a
    ``TIMESTAMP`` column round trips as the correct instant wearing the wrong wall clock,
    and ``.replace(tzinfo=UTC)`` on it silently shifts the value by the local offset.

    That failure is invisible in every aggregate and fatal in an as-of join, where a
    feature timestamp shifted by hours crosses entity timestamps it should not. It is the
    whole argument for keeping epoch milliseconds as ``BIGINT`` in the core.
    """
    table = f"{test_namespace}.tz_probe"
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql(f"CREATE TABLE {table} (event_ts TIMESTAMP, event_ms BIGINT) USING iceberg")

    moment = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)
    spark.createDataFrame(
        [(moment, to_millis(moment))], "event_ts timestamp, event_ms bigint"
    ).writeTo(table).append()
    row = spark.sql(f"SELECT event_ts, event_ms FROM {table}").collect()[0]

    returned = row["event_ts"]
    assert returned.tzinfo is None, "collect() returns a naive datetime"
    assert returned.astimezone(UTC) == moment, "the instant is correct"

    local_offset = datetime.now().astimezone().utcoffset()
    assert local_offset is not None
    if local_offset != timedelta(0):
        # The naive wall clock is local, not UTC, whenever the driver is not on UTC.
        assert returned.replace(tzinfo=UTC) != moment

    # The integer encoding has no such ambiguity, in either direction.
    assert row["event_ms"] == to_millis(moment)

    spark.sql(f"DROP TABLE {table} PURGE")
