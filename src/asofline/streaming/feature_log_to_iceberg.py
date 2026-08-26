"""Kafka to Iceberg: landing the skew detector's own input.

Sibling of ``streaming.to_iceberg``, same shape, different table and a different explode
step. Read that module's docstring first; both of its hard-won lessons apply here
unchanged:

**Sink mechanism: ``foreachBatch`` plus ``writeTo(...).append()``, not
``writeStream.format("iceberg")``.** ``from_json`` always produces nullable struct fields
regardless of the declared JSON schema's nullability, and the native Iceberg streaming
sink's strict compatibility check rejects a nullable column against this table's
``NOT NULL`` columns (``log_id``, ``view_name``, ``entity_keys``, ...). ``foreachBatch``
sidesteps the whole class of problem by hosting the write on ``DataFrame.writeTo(...)``.

**``foreachBatch`` runs on a cloned ``SparkSession``.** Iceberg's per-session table cache
does not see writes made through that clone, so ``run`` calls
``spark.catalog.refreshTable`` on the original session after every non-empty batch.

What differs from the raw-events job is the shape of the conversion: one Kafka message
here is one served feature *vector* (``FeatureLogEntry``, a nested JSON object with a
``features`` map), and the target table (``offline.tables.serving_log_ddl``) is long, one
row per ``(log_id, feature_name)``. This module's whole job is that explode, plus the same
epoch-millis-to-``TIMESTAMP`` conversion the raw-events job does.

**Why ``explode`` on a parsed ``MapType`` column, not ``stack``.** ``offline.tiles.py``
uses ``stack`` to unpivot a *fixed, known-ahead-of-time* set of aggregation columns
(``a0_s0``, ``a1_s0``, ...) that a Spark SQL ``GROUP BY`` produced as separate columns.
Here there is no fixed column set: a view's ``features`` map has as many keys as it has
declared features, and a future view can declare a different set without this job
changing. ``from_json`` already parses a JSON object with unknown/variable keys straight
into a ``MapType`` column when given a ``MapType`` schema (rather than a ``StructType``
with the keys hardcoded), and ``explode`` on a map column is exactly the built-in that
turns each key/value pair into its own row. Reaching for ``stack`` here would mean forcing
the parse through a fixed struct first, which is the one thing this column must not
assume.

**Why ``entity_keys`` parses as ``MapType(StringType(), StringType())``, not a
``StructType``.** Same reasoning: this project's demo views each have exactly one join
key, but a struct schema would hardcode that key's name into the parser, silently breaking
on a future view with a different key. The target table's ``entity_keys`` column is
already ``MAP<STRING, STRING>`` (``offline.tables.SERVING_LOG_COLUMNS``), so the parsed
map passes through unchanged with no reshaping step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from asofline.config import SETTINGS, Settings
from asofline.offline.session import build_session
from asofline.offline.tables import qualified, serving_log_ddl

# Imported rather than duplicated: this job needs the exact same Kafka connector jar as
# streaming.to_iceberg, and a private re-declaration here would drift silently on a
# version bump made only in that module.

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.streaming.query import StreamingQuery

# Distinct from both `asofline-to-iceberg` (raw events) and `asofline-to-redis` (tiles).
# Three consumer groups, on two topics, so a stall in any one is never mistaken for a
# stall in another.
CONSUMER_GROUP = "asofline-feature-log-to-iceberg"

# The wire format is exactly asofline.skew.logging.FeatureLogEntry.to_dict(): epoch
# milliseconds as ints for both timestamps, entity_keys and features as JSON objects with
# view-dependent keys rather than a fixed set of fields.
_FEATURE_LOG_JSON_SCHEMA = StructType(
    [
        StructField("log_id", StringType(), nullable=False),
        StructField("view_name", StringType(), nullable=False),
        StructField("view_version", IntegerType(), nullable=False),
        StructField("entity_keys", MapType(StringType(), StringType()), nullable=False),
        StructField("request_ts_ms", LongType(), nullable=False),
        StructField("served_at_ms", LongType(), nullable=False),
        StructField("features", MapType(StringType(), DoubleType()), nullable=False),
    ]
)


def build_streaming_session(
    app_name: str = "asofline-feature-log-to-iceberg",
    *,
    settings: Settings = SETTINGS,
    driver_memory: str = "2g",
    shuffle_partitions: int = 4,
) -> SparkSession:
    """``build_session``, which already carries the Kafka connector jar.

    See ``offline.session.KAFKA_PACKAGE``'s docstring: every session this project builds
    now carries every jar any Spark test could need, because the JVM-wide SparkSession
    singleton means the jar set is fixed by whichever session in a test process happens to
    be created first, not chosen per job.
    """
    return build_session(
        app_name,
        settings=settings,
        driver_memory=driver_memory,
        shuffle_partitions=shuffle_partitions,
    )


def create_serving_log_table(spark: SparkSession, namespace: str, table: str) -> str:
    """``offline.ingest.create_raw_table``'s pattern, for the serving-log table.

    No direct equivalent exists in ``offline.ingest`` yet, since nothing has needed to
    create this table outside of a Spark job until now.
    """
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
    spark.sql(serving_log_ddl(namespace, table))
    return qualified(namespace, table)


def read_feature_logs(
    spark: SparkSession,
    *,
    settings: Settings = SETTINGS,
    group_id: str = CONSUMER_GROUP,
    starting_offsets: str = "earliest",
) -> DataFrame:
    """The Kafka source, unparsed. See ``streaming.to_iceberg.read_raw_events`` for why
    ``kafka.group.id`` is set here even though Spark's Kafka source manages its own
    progress through its checkpoint rather than through Kafka consumer-group commits: it
    is what gives this job a distinct, named identity in monitoring and ACLs."""
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.feature_log_topic)
        .option("kafka.group.id", group_id)
        .option("startingOffsets", starting_offsets)
        .load()
    )


def parse_and_explode_feature_logs(kafka_df: DataFrame) -> DataFrame:
    """Kafka ``value`` bytes to one row per ``(log_id, feature_name)``, in
    ``SERVING_LOG_COLUMNS`` order.

    ``explode(e.features)`` is the one generator expression in this projection; Spark SQL
    allows a table-generating function directly in a ``selectExpr`` column list as long as
    it is the only one, which is why this stays a single ``selectExpr`` call rather than a
    separate ``explode`` step followed by a rename.
    """
    parsed = kafka_df.select(
        from_json(col("value").cast("string"), _FEATURE_LOG_JSON_SCHEMA).alias("e")
    )
    return parsed.selectExpr(
        "e.log_id AS log_id",
        "e.view_name AS view_name",
        "e.view_version AS view_version",
        "e.entity_keys AS entity_keys",
        "timestamp_millis(e.request_ts_ms) AS request_ts",
        "timestamp_millis(e.served_at_ms) AS served_at_ts",
        "explode(e.features) AS (feature_name, value)",
    )


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

    Mirrors ``streaming.to_iceberg.run`` exactly: ``Trigger.AvailableNow`` (Spark 4.0)
    processes every offset available at query start and exits on its own, which is the
    "drain and stop" shape a bounded test needs, and continuous mode is the same query
    with no trigger override.
    """
    create_serving_log_table(spark, namespace, table)
    target = qualified(namespace, table)
    exploded = parse_and_explode_feature_logs(
        read_feature_logs(
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

    writer = exploded.writeStream.foreachBatch(write_batch).option(
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
        table="serving_log",
        checkpoint_location="checkpoints/feature_log_to_iceberg",
    )
    _query.awaitTermination()
