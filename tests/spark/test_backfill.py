"""P1 done-test: does the tiled Spark backfill agree with a from-scratch recomputation?

The oracle in ``tests/oracle.py`` never touches the tile table, the compiler or Spark. If
these agree, tiling and the two window strategies are faithful refactorings of the
definition rather than a second, differently wrong definition.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from pyspark.sql import SparkSession

from asofline.compiler.batch import Strategy, backfill_features
from asofline.compiler.spec import feature_specs
from asofline.definitions.aggregation import AggFunction
from asofline.definitions.view import FeatureView
from asofline.demo.events import EngagementEvent
from asofline.demo.generator import EngagementGenerator, GeneratorConfig
from asofline.demo.views import USER_ENGAGEMENT, VIDEO_ENGAGEMENT
from asofline.offline.ingest import load_events
from asofline.offline.tiles import build_tiles
from tests.oracle import expected_features, group_by_entity, values_agree

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_backfill_test"
CONFIG = GeneratorConfig(n_events=40_000, n_users=400, n_videos=1_500)
PROBE_ROWS = 150


@dataclass(frozen=True, slots=True)
class Fixture:
    events: list[EngagementEvent]
    raw_table: str


@pytest.fixture(scope="module")
def seeded(spark: SparkSession) -> Iterator[Fixture]:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    events = EngagementGenerator(CONFIG).generate()
    raw = load_events(spark, events, namespace=NAMESPACE, table="engagement_events")
    for view in (USER_ENGAGEMENT, VIDEO_ENGAGEMENT):
        build_tiles(spark, view, namespace=NAMESPACE, raw_table=raw)
    yield Fixture(events=events, raw_table=raw)


def _probe_rows(events: list[EngagementEvent], join_key: str, seed: int) -> list[tuple[str, int]]:
    """Random ``(entity, as_of)`` pairs spread over the whole stream.

    Drawn from the entities that actually appear, so most probes land on an entity with
    history. A uniform draw over the configured population would mostly probe entities the
    Zipf sampler never emitted, and would test the empty case over and over.
    """
    rng = random.Random(seed)
    lo = min(event.event_ts for event in events)
    hi = max(event.event_ts for event in events)
    entities = sorted({getattr(event, join_key) for event in events})
    return [(rng.choice(entities), rng.randrange(lo, hi)) for _ in range(PROBE_ROWS)]


def _run(
    spark: SparkSession, view: FeatureView, fixture: Fixture, strategy: Strategy, seed: int
) -> tuple[list[dict[str, object]], dict[str, list[EngagementEvent]]]:
    join_key = view.join_keys[0]
    rows = _probe_rows(fixture.events, join_key, seed)
    entity_df = spark.createDataFrame(rows, f"{join_key} string, as_of_ms bigint").selectExpr(
        join_key, "timestamp_millis(as_of_ms) AS as_of_ts"
    )
    result = backfill_features(
        spark,
        view,
        entity_df,
        namespace=NAMESPACE,
        raw_table=fixture.raw_table,
        strategy=strategy,
    )
    return [row.asDict() for row in result.collect()], group_by_entity(fixture.events, join_key)


def _mismatches(
    view: FeatureView,
    rows: list[dict[str, object]],
    by_entity: dict[str, list[EngagementEvent]],
) -> list[str]:
    join_key = view.join_keys[0]
    problems: list[str] = []
    for row in rows:
        entity = str(row[join_key])
        as_of_ms = int(row["as_of_ts"].timestamp() * 1000)  # type: ignore[union-attr]
        expected = expected_features(view, by_entity, entity, as_of_ms)
        for name, want in expected.items():
            got = row[name]
            assert got is None or isinstance(got, float | int)
            if not values_agree(want, None if got is None else float(got)):
                problems.append(f"{entity}@{as_of_ms} {name}: expected {want!r}, got {got!r}")
    return problems


@pytest.mark.parametrize("strategy", list(Strategy), ids=str)
@pytest.mark.parametrize("view", [USER_ENGAGEMENT, VIDEO_ENGAGEMENT], ids=lambda v: v.name)
def test_backfill_matches_the_oracle(
    spark: SparkSession, seeded: Fixture, view: FeatureView, strategy: Strategy
) -> None:
    rows, by_entity = _run(spark, view, seeded, strategy, seed=11)
    assert len(rows) == PROBE_ROWS
    problems = _mismatches(view, rows, by_entity)
    assert not problems, f"{len(problems)} mismatches:\n" + "\n".join(problems[:20])


def test_the_video_view_actually_exercises_the_range_path(spark: SparkSession) -> None:
    """Guards the parametrisation above from becoming vacuous.

    ``MAX`` has no inverse, so it takes the range path under either strategy. If a future
    edit dropped it from the demo registry, the PREFIX run would silently stop covering
    the range path and nothing would fail.
    """
    functions = {spec.function for spec in feature_specs(VIDEO_ENGAGEMENT)}
    assert AggFunction.MAX in functions
    assert not AggFunction.MAX.has_inverse


def test_the_two_strategies_agree_to_float_tolerance(spark: SparkSession, seeded: Fixture) -> None:
    """PREFIX subtracts running totals; RANGE folds the covered tiles. Same answer.

    Not bit for bit, and the assertion says so. Float addition is not associative, so the
    two orderings differ in the last bits. Anything comparing them, the skew detector
    included, needs a tolerance for exactly this reason.
    """
    prefix_rows, _ = _run(spark, USER_ENGAGEMENT, seeded, Strategy.PREFIX, seed=23)
    range_rows, _ = _run(spark, USER_ENGAGEMENT, seeded, Strategy.RANGE, seed=23)
    assert len(prefix_rows) == len(range_rows)

    names = [spec.feature_name for spec in feature_specs(USER_ENGAGEMENT)]
    by_id = {row["entity_row_id"]: row for row in range_rows}
    disagreements: list[str] = []
    for left in prefix_rows:
        right = by_id[left["entity_row_id"]]
        for name in names:
            a, b = left[name], right[name]
            a = None if a is None else float(a)  # type: ignore[arg-type]
            b = None if b is None else float(b)  # type: ignore[arg-type]
            if not values_agree(a, b):
                disagreements.append(f"{name}: {a!r} vs {b!r}")
    assert not disagreements, "\n".join(disagreements[:20])


def test_an_entity_with_no_history_gets_zeros_and_nulls_not_missing_rows(
    spark: SparkSession, seeded: Fixture
) -> None:
    """The empty case, which both strategies must answer identically.

    Sum and count of nothing are zero; min, max and mean of nothing are null. Serving a
    null where a zero belongs is the kind of difference a model notices and a spot check
    does not.
    """
    entity_df = spark.createDataFrame(
        [("u_does_not_exist", seeded.events[0].event_ts)], "user_id string, as_of_ms bigint"
    ).selectExpr("user_id", "timestamp_millis(as_of_ms) AS as_of_ts")
    for strategy in Strategy:
        row = (
            backfill_features(
                spark,
                USER_ENGAGEMENT,
                entity_df,
                namespace=NAMESPACE,
                raw_table=seeded.raw_table,
                strategy=strategy,
            )
            .collect()[0]
            .asDict()
        )
        assert row["watch_seconds_sum_1h"] == 0.0, strategy
        assert row["count_1d"] == 0.0, strategy
        assert row["watch_seconds_avg_1d"] is None, strategy
