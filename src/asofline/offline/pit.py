"""Point-in-time correct training data, under two semantics that differ on one predicate.

A backfill answers: what would this entity's feature vector have been at time ``T``? Three
predicates decide which events may contribute, and the third is the one almost everyone
omits.

1. ``event_ts < T``. No values from the future. Everybody gets this right.
2. Freshness. An entity whose newest known event predates ``T - ttl`` is served nulls
   rather than a stale vector, and the same rule applies on both paths.
3. ``created_ts <= T``. Not merely in the past, but **known** by then.

Feast treats ``created_timestamp`` as a tiebreak when deduplicating, not as a filter.
Omitting it is invisible in every test built from data that arrived in order, and wrong on
every real stream, because an event that arrived after ``T`` was not available to a model
serving at ``T`` no matter when it happened.

Hence the two modes:

``EVENT_TIME``
    Predicate 3 dropped. Reads the precomputed tile table, which was binned by event time
    with no regard for arrival, so it is fast. It leaks. It is kept because the size of
    the leak is the most interesting number this project produces, and you cannot measure
    a gap with only one side of it.

``KNOWN``
    All three predicates, computed straight from raw events with no tiles involved,
    because a tile is a function of every event that landed in it and cannot be filtered
    by arrival after the fact. Slower, and correct.

``KNOWN`` is deliberately the obvious implementation rather than a clever one. It is the
reference the fast path is checked against, and being able to read it and see that it is
right is worth more than being able to run it quickly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from asofline.compiler.batch import (
    AS_OF_COLUMN,
    ROW_ID,
    Backfill,
    Strategy,
    backfill_features,
    normalize_entities,
)
from asofline.compiler.spec import FeatureSpec, feature_specs
from asofline.definitions.view import FeatureView

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class Semantics(StrEnum):
    EVENT_TIME = "as_of_event_time"
    KNOWN = "as_of_known"

    @property
    def filters_on_arrival(self) -> bool:
        return self is Semantics.KNOWN


def _keys(view: FeatureView, prefix: str) -> str:
    return ", ".join(f"{prefix}.{key}" for key in view.join_keys)


def _key_join(view: FeatureView, left: str, right: str) -> str:
    return " AND ".join(f"{left}.{key} = {right}.{key}" for key in view.join_keys)


def _arrival_predicate(semantics: Semantics, entity_alias: str) -> str:
    """The one predicate that separates the two modes. Deliberately isolated here.

    Written as a literal ``TRUE`` rather than by assembling the WHERE clause differently,
    so the two modes execute the same query shape and a difference in results cannot be
    blamed on a difference in plan.
    """
    if not semantics.filters_on_arrival:
        return "TRUE"
    return f"unix_millis(r.created_ts) <= {entity_alias}.as_of_ms"


def _value_expr(view: FeatureView, alias: str = "r") -> str:
    """Map a spec's source column name to the raw column, as SQL.

    A ``CASE`` over the column names the view actually declares, so a typo in a definition
    produces no branch and a null rather than a silently wrong column.
    """
    columns = sorted({spec.column for spec in feature_specs(view) if spec.column})
    branches = " ".join(
        f"WHEN '{column}' THEN CAST({alias}.{column} AS DOUBLE)" for column in columns
    )
    # COUNT has no column and is handed a constant, matching Monoid.lift for COUNT
    # ignoring the value it is given.
    return f"CASE f.column_name {branches} ELSE 1.0 END"


def _feature_struct(spec: FeatureSpec) -> str:
    column = f"'{spec.column}'" if spec.column else "CAST(NULL AS STRING)"
    return (
        f"struct('{spec.feature_name}' AS feature_name, '{spec.function}' AS fn, "
        f"{column} AS column_name, CAST({spec.window_ms} AS BIGINT) AS window_ms, "
        f"CAST({spec.granularity_ms} AS BIGINT) AS granularity_ms)"
    )


def known_features(
    spark: SparkSession,
    view: FeatureView,
    entities: DataFrame,
    *,
    raw_table: str,
) -> DataFrame:
    """The reference implementation: all three predicates, straight from raw events.

    The window applies the same snapped trailing edge as the tiled path, so a difference
    between the two is a difference in *which events were visible*, not in what a window
    means. Comparing against an exact ``[T - W, T)`` interval instead would show a gap on
    every row and the gap would be the snapping, drowning the effect being measured.
    """
    specs = feature_specs(view)
    entities.createOrReplaceTempView("_asofline_pit_entities")
    structs = ", ".join(_feature_struct(spec) for spec in specs)
    value = _value_expr(view)

    spark.sql(
        f"""
        SELECT e.{ROW_ID}, {_keys(view, "e")}, e.as_of_ms, f.feature_name, f.fn,
               f.column_name, f.window_ms, f.granularity_ms
        FROM _asofline_pit_entities e
        LATERAL VIEW explode(array({structs})) x AS f
        """
    ).createOrReplaceTempView("_asofline_pit_grid")

    # COUNT(r.event_id) rather than COUNT(*): this is a left join, so a non-matching
    # entity contributes one all-null row and COUNT(*) would report it as one event.
    return spark.sql(
        f"""
        SELECT f.{ROW_ID}, f.feature_name,
               CASE f.fn
                   WHEN 'sum'   THEN COALESCE(SUM({value}), 0.0)
                   WHEN 'count' THEN CAST(COUNT(r.event_id) AS DOUBLE)
                   WHEN 'min'   THEN MIN({value})
                   WHEN 'max'   THEN MAX({value})
                   WHEN 'avg'   THEN SUM({value}) / NULLIF(COUNT({value}), 0)
               END AS value
        FROM _asofline_pit_grid f
        LEFT JOIN {raw_table} r
               ON {_key_join(view, "f", "r")}
              AND unix_millis(r.event_ts)
                    >= ((f.as_of_ms - f.window_ms) DIV f.granularity_ms) * f.granularity_ms
              AND unix_millis(r.event_ts) < f.as_of_ms
              AND {_arrival_predicate(Semantics.KNOWN, "f")}
        GROUP BY f.{ROW_ID}, f.feature_name, f.fn
        """
    )


def entity_freshness(
    spark: SparkSession,
    view: FeatureView,
    entities: DataFrame,
    *,
    raw_table: str,
    semantics: Semantics,
) -> DataFrame:
    """Per entity row: the newest event visible at ``as_of``, and whether it is fresh.

    Predicate 2. Without it, an entity last seen a month ago is served a vector of zeros
    that is indistinguishable from an entity that was active but idle, and a model cannot
    tell "quiet" from "gone".

    The visibility rule here matches the mode. Under ``KNOWN`` an entity whose only events
    all arrived late is correctly treated as not yet seen at all.
    """
    entities.createOrReplaceTempView("_asofline_pit_entities")
    ttl_ms = int(view.ttl.total_seconds() * 1000)
    return spark.sql(
        f"""
        SELECT e.{ROW_ID},
               MAX(unix_millis(r.event_ts)) AS last_known_ms,
               MAX(unix_millis(r.event_ts)) IS NOT NULL
                   AND e.as_of_ms - MAX(unix_millis(r.event_ts)) <= {ttl_ms} AS is_fresh
        FROM _asofline_pit_entities e
        LEFT JOIN {raw_table} r
               ON {_key_join(view, "e", "r")}
              AND unix_millis(r.event_ts) < e.as_of_ms
              AND {_arrival_predicate(semantics, "e")}
        GROUP BY e.{ROW_ID}, e.as_of_ms
        """
    )


def training_features(
    spark: SparkSession,
    view: FeatureView,
    entity_df: DataFrame,
    *,
    namespace: str,
    raw_table: str,
    semantics: Semantics = Semantics.KNOWN,
    strategy: Strategy = Strategy.PREFIX,
    apply_ttl: bool = True,
) -> DataFrame:
    """The public entry point. One frame in, one feature vector per row out."""
    plan = Backfill(view=view, namespace=namespace, raw_table=raw_table, strategy=strategy)
    entities = normalize_entities(spark, plan, entity_df).cache()
    specs = feature_specs(view)

    if semantics is Semantics.EVENT_TIME:
        wide = backfill_features(
            spark,
            view,
            entities,
            namespace=namespace,
            raw_table=raw_table,
            strategy=strategy,
        )
    else:
        long = known_features(spark, view, entities, raw_table=raw_table)
        long.createOrReplaceTempView("_asofline_pit_long")
        entities.createOrReplaceTempView("_asofline_pit_entities")
        pivots = ", ".join(
            f"MAX(CASE WHEN feature_name = '{spec.feature_name}' THEN value END) "
            f"AS {spec.feature_name}"
            for spec in specs
        )
        spark.sql(
            f"""
            SELECT {ROW_ID}, {pivots} FROM _asofline_pit_long GROUP BY {ROW_ID}
            """
        ).createOrReplaceTempView("_asofline_pit_values")
        names = ", ".join(f"v.{spec.feature_name}" for spec in specs)
        wide = spark.sql(
            f"""
            SELECT e.{ROW_ID}, {_keys(view, "e")}, e.{AS_OF_COLUMN}, {names}
            FROM _asofline_pit_entities e
            LEFT JOIN _asofline_pit_values v ON v.{ROW_ID} = e.{ROW_ID}
            """
        )

    if not apply_ttl:
        return wide

    freshness = entity_freshness(spark, view, entities, raw_table=raw_table, semantics=semantics)
    wide.createOrReplaceTempView("_asofline_pit_wide")
    freshness.createOrReplaceTempView("_asofline_pit_freshness")
    gated = ", ".join(
        f"CASE WHEN s.is_fresh THEN w.{spec.feature_name} END AS {spec.feature_name}"
        for spec in specs
    )
    return spark.sql(
        f"""
        SELECT w.{ROW_ID}, {_keys(view, "w")}, w.{AS_OF_COLUMN},
               s.last_known_ms, s.is_fresh, {gated}
        FROM _asofline_pit_wide w
        JOIN _asofline_pit_freshness s ON s.{ROW_ID} = w.{ROW_ID}
        """
    )
