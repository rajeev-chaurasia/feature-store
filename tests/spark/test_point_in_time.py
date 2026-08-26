"""P2 done-test: does a late arrival leak backwards into a closed window?

The adversarial construction is one event that happened before ``T`` but arrived after it.
Under ``KNOWN`` the backfill must return the value it would have returned without that
event, because a model serving at ``T`` could not have seen it. Under ``EVENT_TIME`` it
must not, and that is not a bug being tolerated but the leak being measured.

Asserted on the value, not on a row count. A count assertion passes whether the number is
right or wrong.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from asofline.demo.events import EngagementEvent, EventType, to_millis
from asofline.demo.views import USER_ENGAGEMENT
from asofline.offline.ingest import load_events
from asofline.offline.pit import Semantics, training_features
from asofline.offline.tiles import build_tiles

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_pit_test"
BASE = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
AS_OF = BASE + timedelta(hours=6)
SUBJECT = "u_subject"


def _event(
    event_id: str,
    *,
    happened: datetime,
    arrived: datetime,
    seconds: float,
    user: str = SUBJECT,
) -> EngagementEvent:
    return EngagementEvent(
        event_id=event_id,
        event_type=EventType.WATCH,
        user_id=user,
        video_id="v_subject",
        event_ts=to_millis(happened),
        created_ts=to_millis(arrived),
        watch_seconds=seconds,
    )


# Two events inside the 1h and 24h windows that ended at AS_OF, both promptly delivered.
ON_TIME = [
    _event(
        "on_time_1",
        happened=BASE + timedelta(hours=5, minutes=30),
        arrived=BASE + timedelta(hours=5, minutes=30, seconds=1),
        seconds=10.0,
    ),
    _event(
        "on_time_2",
        happened=BASE + timedelta(hours=5, minutes=50),
        arrived=BASE + timedelta(hours=5, minutes=50, seconds=1),
        seconds=15.0,
    ),
]

# The adversary. It happened ten minutes before AS_OF, so predicate 1 admits it. It
# arrived an hour after AS_OF, so predicate 3 must exclude it.
LATE = _event(
    "the_adversary",
    happened=BASE + timedelta(hours=5, minutes=50),
    arrived=AS_OF + timedelta(hours=1),
    seconds=1_000.0,
)

ON_TIME_TOTAL = 25.0
WITH_LEAK_TOTAL = 1_025.0


@pytest.fixture(scope="module")
def seeded(spark: SparkSession) -> Iterator[str]:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    raw = load_events(spark, [*ON_TIME, LATE], namespace=NAMESPACE, table="engagement_events")
    build_tiles(spark, USER_ENGAGEMENT, namespace=NAMESPACE, raw_table=raw)
    yield raw


def _vector(spark: SparkSession, raw: str, semantics: Semantics) -> dict[str, object]:
    entity_df = spark.createDataFrame(
        [(SUBJECT, to_millis(AS_OF))], "user_id string, as_of_ms bigint"
    ).selectExpr("user_id", "timestamp_millis(as_of_ms) AS as_of_ts")
    rows = training_features(
        spark,
        USER_ENGAGEMENT,
        entity_df,
        namespace=NAMESPACE,
        raw_table=raw,
        semantics=semantics,
    ).collect()
    assert len(rows) == 1
    return rows[0].asDict()


def test_strict_mode_returns_the_value_from_before_the_late_arrival(
    spark: SparkSession, seeded: str
) -> None:
    """The done-test. The old value, to the cent, not merely 'not the new one'."""
    vector = _vector(spark, seeded, Semantics.KNOWN)
    assert vector["watch_seconds_sum_1h"] == pytest.approx(ON_TIME_TOTAL)
    assert vector["watch_seconds_sum_1d"] == pytest.approx(ON_TIME_TOTAL)
    assert vector["count_1h"] == pytest.approx(2.0)
    assert vector["watch_seconds_avg_1d"] == pytest.approx(ON_TIME_TOTAL / 2)


def test_event_time_mode_leaks_the_late_arrival(spark: SparkSession, seeded: str) -> None:
    """The other half of the comparison, asserted so the leak cannot quietly disappear.

    If this ever starts passing strict semantics, the measurement in
    ``test_the_two_modes_disagree`` becomes a comparison of a mode against itself and
    would report a leak of zero, which reads as good news.
    """
    vector = _vector(spark, seeded, Semantics.EVENT_TIME)
    assert vector["watch_seconds_sum_1h"] == pytest.approx(WITH_LEAK_TOTAL)
    assert vector["count_1h"] == pytest.approx(3.0)


def test_the_two_modes_disagree_on_exactly_the_adversary(spark: SparkSession, seeded: str) -> None:
    strict = _vector(spark, seeded, Semantics.KNOWN)
    leaky = _vector(spark, seeded, Semantics.EVENT_TIME)
    difference = float(leaky["watch_seconds_sum_1h"]) - float(strict["watch_seconds_sum_1h"])  # type: ignore[arg-type]
    assert difference == pytest.approx(LATE.watch_seconds)


def test_an_event_that_arrives_exactly_at_as_of_is_visible(spark: SparkSession) -> None:
    """``created_ts <= T`` is inclusive while ``event_ts < T`` is exclusive.

    Not an oversight. An event already recorded at the instant of the query is available
    to it, whereas an event happening at that instant is the thing being predicted. The
    two boundaries genuinely differ, and this pins both.
    """
    namespace = f"{NAMESPACE}_boundary"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
    at_boundary = _event(
        "arrives_at_as_of",
        happened=AS_OF - timedelta(minutes=1),
        arrived=AS_OF,
        seconds=7.0,
    )
    at_as_of = _event("happens_at_as_of", happened=AS_OF, arrived=AS_OF, seconds=99.0)
    raw = load_events(
        spark, [at_boundary, at_as_of], namespace=namespace, table="engagement_events"
    )
    build_tiles(spark, USER_ENGAGEMENT, namespace=namespace, raw_table=raw)

    entity_df = spark.createDataFrame(
        [(SUBJECT, to_millis(AS_OF))], "user_id string, as_of_ms bigint"
    ).selectExpr("user_id", "timestamp_millis(as_of_ms) AS as_of_ts")
    vector = (
        training_features(
            spark,
            USER_ENGAGEMENT,
            entity_df,
            namespace=namespace,
            raw_table=raw,
            semantics=Semantics.KNOWN,
        )
        .collect()[0]
        .asDict()
    )
    assert vector["watch_seconds_sum_1h"] == pytest.approx(7.0)
    assert vector["count_1h"] == pytest.approx(1.0)


def test_a_stale_entity_is_served_nulls_not_zeros(spark: SparkSession) -> None:
    """Predicate 2. 'Quiet' and 'gone' are different, and zeros cannot express the second.

    The view's ttl is seven days, so an entity whose newest visible event is older than
    that reads null everywhere rather than a plausible looking vector of zeros.
    """
    namespace = f"{NAMESPACE}_stale"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
    ancient = _event(
        "long_ago",
        happened=AS_OF - timedelta(days=30),
        arrived=AS_OF - timedelta(days=30),
        seconds=5.0,
    )
    raw = load_events(spark, [ancient], namespace=namespace, table="engagement_events")
    build_tiles(spark, USER_ENGAGEMENT, namespace=namespace, raw_table=raw)

    entity_df = spark.createDataFrame(
        [(SUBJECT, to_millis(AS_OF))], "user_id string, as_of_ms bigint"
    ).selectExpr("user_id", "timestamp_millis(as_of_ms) AS as_of_ts")
    row = (
        training_features(
            spark,
            USER_ENGAGEMENT,
            entity_df,
            namespace=namespace,
            raw_table=raw,
            semantics=Semantics.KNOWN,
        )
        .collect()[0]
        .asDict()
    )
    assert row["is_fresh"] is False
    assert row["watch_seconds_sum_7d"] is None
    assert row["count_1h"] is None
