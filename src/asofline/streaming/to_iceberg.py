"""Kafka to Iceberg: the raw-landing half of P4.

This is one of two independent consumers of ``engagement_events``. The other, a
``confluent-kafka`` consumer that maintains Redis tile state, is built separately behind
its own shared codec contract. The plan is explicit that these must not share a consumer
group: two consumers, one topic, independent lag, so a stall in one is never mistaken for
a stall in the other.

**Sink mechanism: ``foreachBatch`` plus ``writeTo(...).append()``, not
``writeStream.format("iceberg")``.**

Both were tried against this REST catalog. The native Iceberg streaming sink failed with
a real, reproducible schema error: ``from_json`` always produces nullable struct fields
regardless of the declared JSON schema's nullability, and Iceberg's native writer runs a
strict compatibility check that rejects writing a nullable column into a ``NOT NULL``
raw-table column (``event_id``, ``event_ts``, ...). Working around that would mean
loosening the raw table's schema or forcing non-null casts through a code path this
project does not otherwise touch. ``foreachBatch`` sidesteps the whole class of problem by
handing each micro-batch to ``DataFrame.writeTo(...).append()``, the exact call
``offline.ingest.load_events`` already uses for the batch path, so the write side has one
proven implementation instead of two.

One more thing had to be learned the hard way: ``foreachBatch`` executes on a clone of the
driving ``SparkSession``, and Iceberg's per-session table cache does not see writes made
through that clone. A caller holding the original session (every test in this project, and
any long-running process that inspects the table while the stream runs) reads stale
metadata, snapshot history included, until that specific table is refreshed on the
original session. ``run`` does this after every non-empty batch so a caller never has to
know the mechanism exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from asofline.config import SETTINGS, Settings
from asofline.offline.ingest import create_raw_table
from asofline.offline.session import ICEBERG_PACKAGES, build_session
from asofline.offline.tables import RAW_EVENTS_COLUMNS, qualified

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.streaming.query import StreamingQuery

# A name distinct from whatever the Kafka-to-Redis consumer uses, per the plan's "two
# consumers, one topic, independent lag" requirement.
CONSUMER_GROUP = "asofline-to-iceberg"

# build_session() bakes in the Iceberg packages every offline job needs but knows nothing
# about Kafka. Adding this package through the `extra` config rather than editing
# session.py keeps that builder generic; only the streaming jobs pay for the Kafka jars.
_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4"

# The wire format is exactly asofline.demo.events.EngagementEvent.to_dict(): epoch
# milliseconds as ints for both timestamps, watch_seconds nullable, liked/shared always
# present. Field names and nullability here must track that dataclass, not the raw table
# (the table's TIMESTAMP columns and NOT NULL constraints are a separate, later step).
_EVENT_JSON_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=False),
        StructField("event_type", StringType(), nullable=False),
        StructField("user_id", StringType(), nullable=False),
        StructField("video_id", StringType(), nullable=False),
        StructField("event_ts", LongType(), nullable=False),
        StructField("created_ts", LongType(), nullable=False),
        StructField("watch_seconds", DoubleType(), nullable=True),
        StructField("liked", IntegerType(), nullable=False),
        StructField("shared", IntegerType(), nullable=False),
    ]
)

# Columns converted from epoch-ms BIGINT to TIMESTAMP on the way into the raw table.
# Matches offline.ingest.events_dataframe's `timestamp_millis(...)` conversion exactly;
# duplicated rather than imported because that module builds its DataFrame from Python
# objects while this one parses JSON off the wire, so the parsing step cannot literally be
# one function. The column set and order both still come from RAW_EVENTS_COLUMNS, the one
# shared source of truth for the raw table's shape, so the two paths cannot silently drift
# on what the raw table looks like even though they differ on how a row is produced.
_MILLIS_TO_TIMESTAMP = frozenset({"event_ts", "created_ts"})


def build_streaming_session(
    app_name: str = "asofline-to-iceberg",
    *,
    settings: Settings = SETTINGS,
    driver_memory: str = "2g",
    shuffle_partitions: int = 4,
) -> SparkSession:
    """``build_session`` plus the Kafka connector jar this job needs."""
    packages = ",".join((*ICEBERG_PACKAGES, _KAFKA_PACKAGE))
    return build_session(
        app_name,
        settings=settings,
        driver_memory=driver_memory,
        shuffle_partitions=shuffle_partitions,
        extra={"spark.jars.packages": packages},
    )


def read_raw_events(
    spark: SparkSession,
    *,
    settings: Settings = SETTINGS,
    group_id: str = CONSUMER_GROUP,
    starting_offsets: str = "earliest",
) -> DataFrame:
    """The Kafka source, unparsed.

    ``kafka.group.id`` does not drive offset management here: Spark's Kafka source tracks
    progress in its own checkpoint rather than committing offsets under a Kafka consumer
    group. Setting it anyway is what makes this job show up as a distinct, named consumer
    in monitoring and ACLs, separate from the Redis consumer's group, which is the part of
    "independent lag" that is actually about identity rather than mechanism.
    """
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.events_topic)
        .option("kafka.group.id", group_id)
        .option("startingOffsets", starting_offsets)
        .load()
    )


def parse_events(kafka_df: DataFrame) -> DataFrame:
    """Kafka ``value`` bytes to raw-table-shaped columns, in ``RAW_EVENTS_COLUMNS`` order."""
    parsed = kafka_df.select(from_json(col("value").cast("string"), _EVENT_JSON_SCHEMA).alias("e"))
    select_exprs = [
        f"timestamp_millis(e.{name}) AS {name}"
        if name in _MILLIS_TO_TIMESTAMP
        else f"e.{name} AS {name}"
        for name, _ in RAW_EVENTS_COLUMNS
    ]
    return parsed.selectExpr(*select_exprs)


def run(
    spark: SparkSession,
    *,
    namespace: str,
    table: str,
    checkpoint_location: str,
    settings: Settings = SETTINGS,
    group_id: str = CONSUMER_GROUP,
    starting_offsets: str = "earliest",
    bounded: bool = False,
) -> StreamingQuery:
    """Start the stream. ``bounded=True`` drains what is currently in the topic and stops.

    ``Trigger.AvailableNow`` (Spark 4.0) is what makes ``bounded`` meaningful: it processes
    every offset available at query start and then exits on its own, which is the "drain
    and stop" shape a test needs. Continuous mode (``bounded=False``) is the same query
    with no trigger override, so the only difference between a test run and a demo run is
    this one flag.
    """
    create_raw_table(spark, namespace, table)
    target = qualified(namespace, table)
    parsed = parse_events(
        read_raw_events(
            spark, settings=settings, group_id=group_id, starting_offsets=starting_offsets
        )
    )

    def write_batch(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.isEmpty():
            return
        batch_df.writeTo(target).append()
        # See the module docstring: foreachBatch runs on a cloned session, and the
        # original session's Iceberg table cache does not otherwise learn of the write.
        spark.catalog.refreshTable(target)

    writer = parsed.writeStream.foreachBatch(write_batch).option(
        "checkpointLocation", checkpoint_location
    )
    if bounded:
        writer = writer.trigger(availableNow=True)
    query = writer.start()
    if bounded:
        query.awaitTermination()
    return query


if __name__ == "__main__":
    _spark = build_streaming_session()
    _query = run(
        _spark,
        namespace=SETTINGS.catalog_name,
        table="engagement_events",
        checkpoint_location="checkpoints/to_iceberg",
    )
    _query.awaitTermination()
