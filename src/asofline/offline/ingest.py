"""Landing generated events in the raw Iceberg table.

Only used to seed a run. In the deployed shape the raw table is fed by the Structured
Streaming job in ``asofline.streaming``; this is the same table written the short way so
the offline path can be built and measured before the streaming path exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asofline.demo.events import EngagementEvent
from asofline.offline.tables import qualified, raw_events_ddl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# The generator emits epoch milliseconds. Timestamps become TIMESTAMP columns here, inside
# Spark, so no Python datetime is constructed anywhere on this path and the naive-datetime
# trap has nowhere to occur.
_RAW_INPUT_SCHEMA = (
    "event_id string, event_type string, user_id string, video_id string, "
    "event_ms bigint, created_ms bigint, watch_seconds double, liked int, shared int"
)

_TO_TABLE = (
    "event_id",
    "event_type",
    "user_id",
    "video_id",
    "timestamp_millis(event_ms) AS event_ts",
    "timestamp_millis(created_ms) AS created_ts",
    "watch_seconds",
    "liked",
    "shared",
)


def events_dataframe(spark: SparkSession, events: list[EngagementEvent]) -> DataFrame:
    rows = [
        (
            event.event_id,
            str(event.event_type),
            event.user_id,
            event.video_id,
            event.event_ts,
            event.created_ts,
            event.watch_seconds,
            event.liked,
            event.shared,
        )
        for event in events
    ]
    return spark.createDataFrame(rows, _RAW_INPUT_SCHEMA).selectExpr(*_TO_TABLE)


def create_raw_table(spark: SparkSession, namespace: str, table: str) -> str:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
    spark.sql(raw_events_ddl(namespace, table))
    return qualified(namespace, table)


def load_events(
    spark: SparkSession,
    events: list[EngagementEvent],
    *,
    namespace: str,
    table: str,
    replace: bool = True,
) -> str:
    """Write ``events`` to the raw table.

    ``replace`` defaults to true because every measurement in this project is computed
    over a named, seeded stream. Appending a second run onto the first would silently
    produce a stream that no ``GeneratorConfig`` describes, and the committed artifact
    would name a stream that never existed.

    Replace drops the table rather than calling ``overwritePartitions``, which only
    replaces partitions the new data happens to touch. A shorter or later-starting stream
    would leave the previous run's days in place and quietly union the two.
    """
    if replace:
        spark.sql(f"DROP TABLE IF EXISTS {qualified(namespace, table)} PURGE")
    name = create_raw_table(spark, namespace, table)
    events_dataframe(spark, events).sortWithinPartitions("event_ts").writeTo(name).append()
    return name
