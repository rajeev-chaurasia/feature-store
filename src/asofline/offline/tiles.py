"""Building the tile table from raw events.

One pass per grid, not one per aggregation. Every aggregation that resolves to the same
granularity shares a single ``GROUP BY (keys, tile_index)``, and the wide result is
unpivoted into the long tile schema with ``stack``. For the demo user view that is two
scans instead of six.

This job implements the ``as_of_event_time`` half of the project on purpose: it bins every
event by ``event_ts`` and ignores ``created_ts`` entirely, so a late arrival retroactively
changes a tile it should not have been able to reach. That is the leak P2 measures. The
strict path does not use these tiles at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asofline.compiler.spec import TileSpec, tile_specs
from asofline.definitions.aggregation import AggFunction
from asofline.definitions.view import FeatureView
from asofline.offline.tables import qualified, tile_columns, tile_table_ddl, tile_table_name

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

_NULL = "CAST(NULL AS DOUBLE)"


def _as_double(expression: str) -> str:
    """Every state column is DOUBLE, because the tile schema has exactly two of them.

    Not cosmetic. ``SUM`` over an INT column returns BIGINT, so an indicator column like
    ``liked`` produced a BIGINT state while ``watch_seconds`` produced a DOUBLE, and the
    ``stack`` unpivot rejected the mismatch. Casting at the source keeps the widening in
    one place instead of at every call site.
    """
    return f"CAST({expression} AS DOUBLE)"


def state_expressions(spec: TileSpec) -> tuple[str, str]:
    """The ``(s0, s1)`` aggregate expressions for one tile spec.

    Null handling is load-bearing and follows plain SQL. ``sum`` and ``avg`` skip nulls,
    so ``watch_seconds_avg`` is seconds per watch rather than seconds per event, with no
    filter language needed in the definition layer. ``count`` counts rows, including the
    ones whose measure column is null, because it is counting events.
    """
    column = spec.column
    match spec.function:
        case AggFunction.SUM:
            return _as_double(f"SUM({column})"), _NULL
        case AggFunction.COUNT:
            return _as_double("COUNT(*)"), _NULL
        case AggFunction.MIN:
            return _as_double(f"MIN({column})"), _NULL
        case AggFunction.MAX:
            return _as_double(f"MAX({column})"), _NULL
        case AggFunction.AVG:
            return _as_double(f"SUM({column})"), _as_double(f"COUNT({column})")
    raise ValueError(f"no state expression for {spec.function}")  # pragma: no cover


def _tile_index_expr(granularity_ms: int) -> str:
    """Integer division, matching ``agg.window.tile_index``.

    ``DIV`` truncates toward zero rather than flooring, which differs from the Python
    implementation for timestamps before 1970. Event timestamps here are always after the
    epoch, and ``assert_epoch_positive`` below checks that rather than assuming it.
    """
    return f"(unix_millis(event_ts) DIV {granularity_ms})"


def assert_epoch_positive(spark: SparkSession, raw_table: str) -> None:
    negative = spark.sql(
        f"SELECT COUNT(*) AS n FROM {raw_table} WHERE unix_millis(event_ts) < 0"
    ).collect()[0]["n"]
    if negative:
        raise ValueError(
            f"{negative} rows in {raw_table} predate the epoch. Tile indexing uses DIV, "
            f"which truncates toward zero and would fold the tiles either side of 1970 "
            f"together. Switch _tile_index_expr to FLOOR before ingesting such data."
        )


def _grid_frame(
    spark: SparkSession,
    view: FeatureView,
    raw_table: str,
    granularity_ms: int,
    specs: tuple[TileSpec, ...],
) -> DataFrame:
    keys = ", ".join(view.join_keys)
    index = _tile_index_expr(granularity_ms)

    wide_columns: list[str] = []
    stack_parts: list[str] = []
    for position, spec in enumerate(specs):
        s0, s1 = state_expressions(spec)
        wide_columns.append(f"{s0} AS a{position}_s0")
        wide_columns.append(f"{s1} AS a{position}_s1")
        stack_parts.append(f"'{spec.agg_name}', a{position}_s0, a{position}_s1")

    wide = spark.sql(
        f"""
        SELECT {keys},
               {index} AS tile_index,
               {", ".join(wide_columns)}
        FROM {raw_table}
        GROUP BY {keys}, {index}
        """
    )
    wide.createOrReplaceTempView("_asofline_wide")
    long = spark.sql(
        f"""
        SELECT {keys},
               tile_index,
               stack({len(specs)}, {", ".join(stack_parts)}) AS (agg_name, s0, s1)
        FROM _asofline_wide
        """
    )
    long.createOrReplaceTempView("_asofline_long")
    # A null s0 means nothing contributed to this tile for this aggregation, which merges
    # to the identity and is therefore indistinguishable from an absent tile. Not writing
    # it keeps the table proportional to real activity rather than to the cross product of
    # entities and aggregations.
    return spark.sql(
        f"""
        SELECT {keys},
               agg_name,
               CAST({granularity_ms} AS BIGINT) AS granularity_ms,
               CAST(tile_index AS BIGINT) AS tile_index,
               timestamp_millis(tile_index * {granularity_ms}) AS tile_start,
               s0,
               s1
        FROM _asofline_long
        WHERE s0 IS NOT NULL
        """
    )


def build_tiles(
    spark: SparkSession,
    view: FeatureView,
    *,
    namespace: str,
    raw_table: str,
    rebuild: bool = True,
) -> str:
    """Materialise every tile for ``view`` and return the tile table name."""
    assert_epoch_positive(spark, raw_table)
    name = qualified(namespace, tile_table_name(view))
    if rebuild:
        spark.sql(f"DROP TABLE IF EXISTS {name} PURGE")
    spark.sql(tile_table_ddl(namespace, view))

    specs = tile_specs(view)
    by_grid: dict[int, list[TileSpec]] = {}
    for spec in specs:
        by_grid.setdefault(spec.granularity_ms, []).append(spec)

    columns = list(tile_columns(view))
    for granularity_ms in sorted(by_grid):
        frame = _grid_frame(spark, view, raw_table, granularity_ms, tuple(by_grid[granularity_ms]))
        frame.select(*columns).sortWithinPartitions("tile_start").writeTo(name).append()
    return name
