"""Iceberg table shapes, derived from feature definitions rather than hand written.

One decision governs this module. **Timestamps are stored as ``TIMESTAMP`` and converted
to epoch milliseconds with ``unix_millis`` at the boundary into the pure core**, rather
than stored as ``BIGINT`` and converted back for partitioning.

Storing millis directly was the first instinct, since the core is millis everywhere. It
loses partition pruning: Iceberg's ``days()`` transform needs a date or timestamp column,
so a millis column would need a duplicated partition column beside it, and two columns
that must agree is a drift risk for the sake of avoiding a cast.

``unix_millis`` is safe where the naive-datetime trap of
``tests/spark/test_smoke_iceberg.py`` is not: it runs inside Spark SQL and returns the
instant, with no session or driver time zone involved. The trap is confined to PySpark's
``collect()``, and nothing here collects a timestamp.
"""

from __future__ import annotations

from asofline.definitions.entity import KeyType
from asofline.definitions.view import FeatureView

_SQL_KEY_TYPES = {KeyType.STRING: "STRING", KeyType.INT64: "BIGINT"}

RAW_EVENTS_COLUMNS = (
    ("event_id", "STRING NOT NULL"),
    ("event_type", "STRING NOT NULL"),
    ("user_id", "STRING NOT NULL"),
    ("video_id", "STRING NOT NULL"),
    ("event_ts", "TIMESTAMP NOT NULL"),
    ("created_ts", "TIMESTAMP NOT NULL"),
    ("watch_seconds", "DOUBLE"),
    ("liked", "INT"),
    ("shared", "INT"),
)

MAX_STATE_ARITY = 2
"""Every supported monoid state fits in two doubles. AVG is the only one that needs both."""


def qualified(namespace: str, table: str) -> str:
    return f"{namespace}.{table}"


def raw_events_ddl(namespace: str, table: str) -> str:
    """Partitioned on event day, not creation day.

    Almost every read filters on event time, including both point-in-time modes. The
    ``created_ts <= T`` predicate that ``as_of_known`` adds prunes nothing and is a
    residual scan inside the surviving partitions, which is stated here so it is not
    later mistaken for a bug.
    """
    columns = ",\n            ".join(f"{name} {sql}" for name, sql in RAW_EVENTS_COLUMNS)
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified(namespace, table)} (
            {columns}
        ) USING iceberg
        PARTITIONED BY (days(event_ts))
        TBLPROPERTIES ('write.parquet.compression-codec' = 'zstd')
    """


def tile_table_name(view: FeatureView) -> str:
    return f"{view.name}__tiles_v{view.version}"


def tile_table_ddl(namespace: str, view: FeatureView) -> str:
    """One tile table per view, keyed by the view's own join keys.

    Per view rather than one global table because the join keys differ between views, and
    a single table would have to carry every view's keys with most of them null.

    ``granularity_ms`` is a column rather than another table, because the two grids share
    every other column and a view's read touches exactly one grid per window, which
    partitioning on granularity makes a pruned scan.
    """
    keys = ",\n            ".join(
        f"{entity.join_key} {_SQL_KEY_TYPES[entity.key_type]} NOT NULL" for entity in view.entities
    )
    states = ",\n            ".join(f"s{i} DOUBLE" for i in range(MAX_STATE_ARITY))
    return f"""
        CREATE TABLE IF NOT EXISTS {qualified(namespace, tile_table_name(view))} (
            {keys},
            agg_name STRING NOT NULL,
            granularity_ms BIGINT NOT NULL,
            tile_index BIGINT NOT NULL,
            tile_start TIMESTAMP NOT NULL,
            {states}
        ) USING iceberg
        PARTITIONED BY (granularity_ms, days(tile_start))
        TBLPROPERTIES ('write.parquet.compression-codec' = 'zstd')
    """


def tile_columns(view: FeatureView) -> tuple[str, ...]:
    """The tile table's columns, in DDL order. Writers select in exactly this order."""
    return (
        *view.join_keys,
        "agg_name",
        "granularity_ms",
        "tile_index",
        "tile_start",
        *(f"s{i}" for i in range(MAX_STATE_ARITY)),
    )
