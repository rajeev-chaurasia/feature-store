"""Compiling a feature view into a Spark backfill over the tile table.

Given an entity frame of ``(join keys, as_of)`` rows, produce the feature vector each row
would have been served at its own timestamp. Three pieces combine:

1. **Whole tiles**, by one of two strategies (below).
2. **The head**, ``[align_down(as_of, g), as_of)``, read from raw events so a recent event
   is not rounded away. This is what makes the leading edge exact.
3. **A merge and finalize** using the same monoid the online path uses.

Two strategies for the whole-tile part, chosen by the algebra:

``PREFIX`` (default, for ``SUM``/``COUNT``/``AVG``)
    These form a group, so a window is ``prefix(end) - prefix(start)``. Getting
    ``prefix(k)`` for an arbitrary ``k`` over sparsely populated tile indices is itself an
    as-of lookup, done here by unioning probe rows into the tile stream and carrying the
    running total forward with ``last(..., ignoreNulls=True)``. One sort, no range join,
    and every feature of every entity is answered in the same pass.

``RANGE`` (required for ``MIN``/``MAX``, available for the rest)
    Join each entity row to the tiles its window covers and aggregate. No inverse needed.
    More work, but it folds the tiles in window order rather than subtracting two running
    totals.

**The two strategies do not agree bit for bit, and that is expected.** Float addition is
not associative, so ``prefix(end) - prefix(start)`` is not the same sequence of operations
as summing the covered tiles. The difference is at the level of accumulated rounding, not
of semantics, and ``RANGE`` exists partly so the gap can be measured rather than assumed
small. Anything comparing the two paths, including the skew detector, needs a tolerance
for this reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from asofline.compiler.spec import FeatureSpec, TileSpec, feature_specs, tile_specs
from asofline.definitions.aggregation import AggFunction
from asofline.definitions.view import FeatureView
from asofline.offline.tables import qualified, tile_table_name

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class Strategy(StrEnum):
    PREFIX = "prefix"
    RANGE = "range"


AS_OF_COLUMN = "as_of_ts"
ROW_ID = "entity_row_id"


@dataclass(frozen=True, slots=True)
class Backfill:
    view: FeatureView
    namespace: str
    raw_table: str
    strategy: Strategy = Strategy.PREFIX

    @property
    def tile_table(self) -> str:
        return qualified(self.namespace, tile_table_name(self.view))

    @property
    def keys(self) -> tuple[str, ...]:
        return self.view.join_keys


def _key_list(keys: tuple[str, ...], prefix: str = "") -> str:
    dot = f"{prefix}." if prefix else ""
    return ", ".join(f"{dot}{key}" for key in keys)


def _key_join(keys: tuple[str, ...], left: str, right: str) -> str:
    return " AND ".join(f"{left}.{key} = {right}.{key}" for key in keys)


def _uses_prefix(spec: FeatureSpec, strategy: Strategy) -> bool:
    return strategy is Strategy.PREFIX and spec.function.has_inverse


def _finalize_expr(spec: FeatureSpec, s0: str, s1: str) -> str:
    """Reconstruct the served value from a merged state, mirroring ``Monoid.finalize``.

    Kept as SQL rather than a UDF: a Python UDF here would serialise every row through
    the driver and would also be a second implementation of finalize that could drift
    from the one the online path calls.

    Both arguments are parenthesised before use. They arrive as composite expressions such
    as ``COALESCE(a, 0.0) + COALESCE(b, 0.0)``, and interpolating one unparenthesised into
    a division silently reassociates it into ``a + (b / c) + d``.
    """
    s0 = f"({s0})"
    s1 = f"({s1})"
    match spec.function:
        case AggFunction.SUM | AggFunction.COUNT:
            # Absent tiles and an empty head mean nothing contributed, and the sum or
            # count of nothing is zero, matching finalize(identity) exactly.
            return f"COALESCE({s0}, 0.0)"
        case AggFunction.MIN | AggFunction.MAX:
            return s0
        case AggFunction.AVG:
            return f"CASE WHEN COALESCE({s1}, 0.0) > 0 THEN {s0} / {s1} ELSE NULL END"
    raise ValueError(f"no finalize for {spec.function}")  # pragma: no cover


def normalize_entities(spark: SparkSession, backfill: Backfill, entity_df: DataFrame) -> DataFrame:
    """Attach a row id and an integer as-of.

    If the caller already supplies ``entity_row_id`` it is used as is. Otherwise one is
    generated with ``row_number`` over a total ordering, which is deterministic but has a
    real cost: an unpartitioned window moves every entity row through a single partition,
    and Spark warns about it. That is acceptable for the entity frames this project
    measures, in the thousands to low millions, and the escape hatch for anything larger
    is to pass an id in.

    ``monotonically_increasing_id`` would avoid the shuffle and is the usual advice, but it
    depends on how the input happens to be partitioned, so recomputing the same frame can
    renumber it. Every number this project commits has to survive being recomputed.
    """
    keys = backfill.keys
    entity_df.createOrReplaceTempView("_asofline_entity_input")
    if ROW_ID in entity_df.columns:
        row_id = ROW_ID
    else:
        row_id = f"ROW_NUMBER() OVER (ORDER BY {_key_list(keys)}, {AS_OF_COLUMN}) AS {ROW_ID}"
    return spark.sql(
        f"""
        SELECT {row_id},
               {_key_list(keys)},
               unix_millis({AS_OF_COLUMN}) AS as_of_ms,
               {AS_OF_COLUMN}
        FROM _asofline_entity_input
        """
    )


def _probe_struct(spec: FeatureSpec, *, sign: int) -> str:
    """One probe: where to read the running total, and with which sign.

    ``+1`` at ``tile_end_index`` and ``-1`` at ``tile_start_index``, so summing the two
    carried totals per feature performs the group subtraction.
    """
    g = spec.granularity_ms
    index = f"(as_of_ms DIV {g})" if sign > 0 else f"((as_of_ms - {spec.window_ms}) DIV {g})"
    return (
        f"struct('{spec.agg_name}' AS agg_name, CAST({g} AS BIGINT) AS granularity_ms, "
        f"CAST({index} AS BIGINT) AS probe_index, CAST({sign} AS DOUBLE) AS sign, "
        f"'{spec.feature_name}' AS feature_name)"
    )


def _prefix_values(
    spark: SparkSession, backfill: Backfill, entities: DataFrame, specs: tuple[FeatureSpec, ...]
) -> DataFrame | None:
    """Whole-tile states for every group-algebra feature, in one union-window pass."""
    if not specs:
        return None
    keys = backfill.keys
    entities.createOrReplaceTempView("_asofline_entities")

    probes = ", ".join(_probe_struct(spec, sign=sign) for spec in specs for sign in (1, -1))
    spark.sql(
        f"""
        SELECT e.{ROW_ID}, {_key_list(keys, "e")}, p.agg_name, p.granularity_ms,
               p.probe_index, p.sign, p.feature_name
        FROM _asofline_entities e
        LATERAL VIEW explode(array({probes})) t AS p
        """
    ).createOrReplaceTempView("_asofline_probes")

    # Probes sort before tiles at the same index, because prefix(k) must exclude the tile
    # at k itself. Getting this backwards double counts one tile per window edge, which is
    # a small error that only appears when as_of lands exactly on a boundary.
    spark.sql(
        f"""
        SELECT {_key_list(keys)}, agg_name, granularity_ms, tile_index AS idx,
               1 AS is_tile, CAST(NULL AS BIGINT) AS {ROW_ID},
               CAST(NULL AS DOUBLE) AS sign, CAST(NULL AS STRING) AS feature_name,
               s0, s1
        FROM {backfill.tile_table}
        UNION ALL
        SELECT {_key_list(keys)}, agg_name, granularity_ms, probe_index AS idx,
               0 AS is_tile, {ROW_ID}, sign, feature_name,
               CAST(NULL AS DOUBLE) AS s0, CAST(NULL AS DOUBLE) AS s1
        FROM _asofline_probes
        """
    ).createOrReplaceTempView("_asofline_stream")

    running = (
        f"OVER (PARTITION BY {_key_list(keys)}, agg_name, granularity_ms "
        f"ORDER BY idx, is_tile ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
    )
    return spark.sql(
        f"""
        SELECT {ROW_ID}, feature_name,
               SUM(sign * COALESCE(cumulative_s0, 0.0)) AS tile_s0,
               SUM(sign * COALESCE(cumulative_s1, 0.0)) AS tile_s1
        FROM (
            SELECT {ROW_ID}, feature_name, sign, is_tile,
                   SUM(s0) {running} AS cumulative_s0,
                   SUM(s1) {running} AS cumulative_s1
            FROM _asofline_stream
        )
        WHERE is_tile = 0
        GROUP BY {ROW_ID}, feature_name
        """
    )


def _range_values(
    spark: SparkSession, backfill: Backfill, entities: DataFrame, specs: tuple[FeatureSpec, ...]
) -> DataFrame | None:
    """Whole-tile states by joining each entity row to the tiles its window covers."""
    if not specs:
        return None
    keys = backfill.keys
    entities.createOrReplaceTempView("_asofline_entities")

    windows = ", ".join(
        f"struct('{spec.agg_name}' AS agg_name, CAST({spec.granularity_ms} AS BIGINT) AS "
        f"granularity_ms, CAST(((as_of_ms - {spec.window_ms}) DIV {spec.granularity_ms}) AS "
        f"BIGINT) AS lo, CAST((as_of_ms DIV {spec.granularity_ms}) AS BIGINT) AS hi, "
        f"'{spec.feature_name}' AS feature_name, '{spec.function}' AS fn)"
        for spec in specs
    )
    spark.sql(
        f"""
        SELECT e.{ROW_ID}, {_key_list(keys, "e")}, w.agg_name, w.granularity_ms,
               w.lo, w.hi, w.feature_name, w.fn
        FROM _asofline_entities e
        LATERAL VIEW explode(array({windows})) t AS w
        """
    ).createOrReplaceTempView("_asofline_ranges")

    # hi is exclusive: it is the head tile's index, and the head is read from raw events.
    return spark.sql(
        f"""
        SELECT r.{ROW_ID}, r.feature_name,
               CASE r.fn
                   WHEN 'min' THEN MIN(t.s0)
                   WHEN 'max' THEN MAX(t.s0)
                   ELSE SUM(t.s0)
               END AS tile_s0,
               SUM(t.s1) AS tile_s1
        FROM _asofline_ranges r
        JOIN {backfill.tile_table} t
          ON {_key_join(keys, "r", "t")}
         AND t.agg_name = r.agg_name
         AND t.granularity_ms = r.granularity_ms
         AND t.tile_index >= r.lo
         AND t.tile_index < r.hi
        GROUP BY r.{ROW_ID}, r.feature_name, r.fn
        """
    )


def _head_state_expr(spec: TileSpec, prefix: str = "r") -> tuple[str, str]:
    """Aggregate expressions for the head, matching ``offline.tiles.state_expressions``.

    Deliberately the same shapes. If the head were aggregated differently from the tiles,
    an event would change value as it crossed a tile boundary, which is a skew source that
    moves with the clock and would be nearly impossible to attribute.
    """
    column = f"{prefix}.{spec.column}" if spec.column else None
    match spec.function:
        case AggFunction.SUM:
            return f"CAST(SUM({column}) AS DOUBLE)", "CAST(NULL AS DOUBLE)"
        case AggFunction.COUNT:
            return "CAST(COUNT(*) AS DOUBLE)", "CAST(NULL AS DOUBLE)"
        case AggFunction.MIN:
            return f"CAST(MIN({column}) AS DOUBLE)", "CAST(NULL AS DOUBLE)"
        case AggFunction.MAX:
            return f"CAST(MAX({column}) AS DOUBLE)", "CAST(NULL AS DOUBLE)"
        case AggFunction.AVG:
            return f"CAST(SUM({column}) AS DOUBLE)", f"CAST(COUNT({column}) AS DOUBLE)"
    raise ValueError(f"no head expression for {spec.function}")  # pragma: no cover


def _head_values(
    spark: SparkSession, backfill: Backfill, entities: DataFrame, specs: tuple[TileSpec, ...]
) -> DataFrame | None:
    """The exact head, ``[align_down(as_of, g), as_of)``, straight from raw events.

    One join per grid, not per feature, since the head interval depends only on ``as_of``
    and the granularity. The join is bounded by one tile's width, so it stays small even
    though it touches the raw table.
    """
    if not specs:
        return None
    keys = backfill.keys
    entities.createOrReplaceTempView("_asofline_entities")

    by_grid: dict[int, list[TileSpec]] = {}
    for spec in specs:
        by_grid.setdefault(spec.granularity_ms, []).append(spec)

    frames: list[DataFrame] = []
    for granularity_ms, grid_specs in sorted(by_grid.items()):
        columns: list[str] = []
        stack_parts: list[str] = []
        for position, spec in enumerate(grid_specs):
            s0, s1 = _head_state_expr(spec)
            columns.append(f"{s0} AS h{position}_s0")
            columns.append(f"{s1} AS h{position}_s1")
            stack_parts.append(f"'{spec.agg_name}', h{position}_s0, h{position}_s1")

        spark.sql(
            f"""
            SELECT e.{ROW_ID}, {", ".join(columns)}
            FROM _asofline_entities e
            JOIN {backfill.raw_table} r
              ON {_key_join(keys, "e", "r")}
             AND unix_millis(r.event_ts) >= (e.as_of_ms DIV {granularity_ms}) * {granularity_ms}
             AND unix_millis(r.event_ts) < e.as_of_ms
            GROUP BY e.{ROW_ID}
            """
        ).createOrReplaceTempView("_asofline_head_wide")
        frames.append(
            spark.sql(
                f"""
                SELECT {ROW_ID}, CAST({granularity_ms} AS BIGINT) AS granularity_ms,
                       stack({len(grid_specs)}, {", ".join(stack_parts)})
                           AS (agg_name, head_s0, head_s1)
                FROM _asofline_head_wide
                """
            )
        )

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def backfill_features(
    spark: SparkSession,
    view: FeatureView,
    entity_df: DataFrame,
    *,
    namespace: str,
    raw_table: str,
    strategy: Strategy = Strategy.PREFIX,
) -> DataFrame:
    """Point-in-time feature values, one column per feature, under event-time semantics.

    **This is the leaky mode.** It reads precomputed tiles, which were binned by
    ``event_ts`` with no regard for when each event actually arrived, so a late arrival
    contributes to a window that closed before it was known. That is deliberate: it is the
    ``as_of_event_time`` half of the comparison P2 measures. The strict path does not go
    through this function.
    """
    plan = Backfill(view=view, namespace=namespace, raw_table=raw_table, strategy=strategy)
    entities = normalize_entities(spark, plan, entity_df).cache()

    specs = feature_specs(view)
    prefix_specs = tuple(spec for spec in specs if _uses_prefix(spec, strategy))
    range_specs = tuple(spec for spec in specs if not _uses_prefix(spec, strategy))

    tile_frames = [
        frame
        for frame in (
            _prefix_values(spark, plan, entities, prefix_specs),
            _range_values(spark, plan, entities, range_specs),
        )
        if frame is not None
    ]
    tiles = tile_frames[0]
    for frame in tile_frames[1:]:
        tiles = tiles.unionByName(frame)
    tiles.createOrReplaceTempView("_asofline_tile_values")

    head = _head_values(spark, plan, entities, tile_specs(view))
    if head is None:  # pragma: no cover - a view with no aggregations cannot be built
        raise ValueError(f"{view.name} produced no tile specs")
    head.createOrReplaceTempView("_asofline_head_values")

    # The driving relation is the full (entity row, feature) grid, built from the specs
    # rather than from whatever the tile queries happened to return. Without it the RANGE
    # strategy loses any (row, feature) pair whose window covers no tiles, and a sum over
    # an empty window would be served as null under RANGE and as zero under PREFIX. The
    # two strategies have to be interchangeable to be comparable.
    # Registered explicitly rather than relying on _head_values having done it.
    entities.createOrReplaceTempView("_asofline_entities")
    grid = ", ".join(
        f"struct('{spec.feature_name}' AS feature_name, '{spec.agg_name}' AS agg_name, "
        f"CAST({spec.granularity_ms} AS BIGINT) AS granularity_ms)"
        for spec in specs
    )
    spark.sql(
        f"""
        SELECT e.{ROW_ID}, f.feature_name, f.agg_name, f.granularity_ms
        FROM _asofline_entities e
        LATERAL VIEW explode(array({grid})) x AS f
        """
    ).createOrReplaceTempView("_asofline_grid")

    spark.sql(
        f"""
        SELECT g.{ROW_ID}, g.feature_name,
               t.tile_s0, t.tile_s1, h.head_s0, h.head_s1
        FROM _asofline_grid g
        LEFT JOIN _asofline_tile_values t
               ON t.{ROW_ID} = g.{ROW_ID} AND t.feature_name = g.feature_name
        LEFT JOIN _asofline_head_values h
               ON h.{ROW_ID} = g.{ROW_ID}
              AND h.agg_name = g.agg_name
              AND h.granularity_ms = g.granularity_ms
        """
    ).createOrReplaceTempView("_asofline_merged")

    merged = {
        AggFunction.MIN: ("LEAST(tile_s0, head_s0)", "COALESCE(tile_s0, head_s0)"),
        AggFunction.MAX: ("GREATEST(tile_s0, head_s0)", "COALESCE(tile_s0, head_s0)"),
    }
    pivots: list[str] = []
    for spec in specs:
        if spec.function in merged:
            both, either = merged[spec.function]
            s0 = (
                f"CASE WHEN tile_s0 IS NOT NULL AND head_s0 IS NOT NULL THEN {both} "
                f"ELSE {either} END"
            )
        else:
            s0 = "COALESCE(tile_s0, 0.0) + COALESCE(head_s0, 0.0)"
        s1 = "COALESCE(tile_s1, 0.0) + COALESCE(head_s1, 0.0)"
        value = _finalize_expr(spec, s0, s1)
        pivots.append(
            f"MAX(CASE WHEN feature_name = '{spec.feature_name}' THEN {value} END) "
            f"AS {spec.feature_name}"
        )

    values = spark.sql(
        f"""
        SELECT {ROW_ID}, {", ".join(pivots)}
        FROM _asofline_merged
        GROUP BY {ROW_ID}
        """
    )
    values.createOrReplaceTempView("_asofline_values")
    entities.createOrReplaceTempView("_asofline_entities")

    names = ", ".join(f"v.{spec.feature_name}" for spec in specs)
    return spark.sql(
        f"""
        SELECT e.{ROW_ID}, {_key_list(view.join_keys, "e")}, e.{AS_OF_COLUMN}, {names}
        FROM _asofline_entities e
        LEFT JOIN _asofline_values v ON v.{ROW_ID} = e.{ROW_ID}
        """
    )
