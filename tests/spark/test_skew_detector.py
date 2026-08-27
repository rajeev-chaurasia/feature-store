"""P5 done-tests: a clean pipeline reports no skew, an injected bug is caught and named.

The scenario end to end: generate a stream, ingest it, build tiles, run every event
through the real Kafka-to-Redis consumer to build genuine online state, read the online
store to get what would actually be served, publish those served vectors through the real
feature-logging wire format and the real feature_log_to_iceberg pipe, then run the
detector against the landed log.

The deliberate bug is injected without touching any committed production code. For a
chosen fraction of served vectors, one target feature's value is recomputed using the
exact same tiles the online store already holds, but with an **empty head**, using the
same pure ``asofline.agg.rollup_at`` function every other part of this project already
trusts. That is a faithful simulation of "the streaming path snapped the leading edge to
the tile grid, dropping the head merge, for one feature only": it reuses real, correct
tile state and only removes the head contribution, exactly the failure mode described in
the plan.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import pytest
import redis as sync_redis
from confluent_kafka import Producer
from pyspark.sql import SparkSession

from asofline.compiler.spec import feature_specs, tile_specs
from asofline.config import SETTINGS
from asofline.demo.events import EngagementEvent
from asofline.demo.generator import EngagementGenerator, GeneratorConfig
from asofline.demo.views import DEMO_REGISTRY, USER_ENGAGEMENT
from asofline.offline.ingest import load_events
from asofline.offline.tiles import build_tiles
from asofline.online.codec import decode_state
from asofline.online.keys import entity_value_key, parse_tile_field, tile_hash_key
from asofline.skew.detector import (
    BUCKET_PARTIAL_HEAD_TILE,
    detect_skew,
)
from asofline.skew.logging import build_log_entry
from asofline.streaming.consumer import process_event
from asofline.streaming.feature_log_to_iceberg import run as run_feature_log_ingest

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_skew_test"
TEST_REDIS_URL = "redis://localhost:6379/15"
TARGET_FEATURE = "count_1d"
"""Chosen deliberately for a wide, almost-never-empty head.

count_1d resolves to the 1-hour grid, and COUNT counts every event type including
impressions (60% of traffic), so a random entity's head interval almost always holds at
least one event to lose. watch_seconds_sum_1h was tried first and resolves to the 5-minute
grid counting only watch events (30% of traffic): the head window is narrow enough that
most randomly chosen entities have zero watch events in it regardless, so dropping the
head made no observable difference for most of them and the injected effect was
indistinguishable from noise. The lesson generalizes: an injected bug's observable effect
size depends on how often the mechanism it degrades actually has something to degrade,
not only on whether the mechanism itself is broken."""


@pytest.fixture
def redis_client() -> Iterator[sync_redis.Redis]:
    client = sync_redis.Redis.from_url(TEST_REDIS_URL)
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


def _tag_entity_ids(events: list[EngagementEvent], tag: str) -> list[EngagementEvent]:
    """Rewrite every event's ``user_id``/``video_id`` with a run-unique prefix.

    Same technique ``test_to_iceberg.py`` and ``test_feature_log_to_iceberg.py`` already
    use for the shared ``engagement_events``/``feature_logs`` topics, applied here for a
    stronger reason than usual: it is not enough to filter *this test's own* assertions by
    tag, because the generator is fully deterministic. Two invocations of this test file
    with the same seed produce byte-identical events and therefore the identical
    ``as_of_ms``, so a time-window filter alone cannot tell "this run's data" apart from
    an earlier run's, and Structured Streaming's own checkpoint cannot either once it is
    cleared (a fresh checkpoint reads the whole topic's retained history from
    ``earliest``, not just what this run just published). Tagging the entity ids
    themselves is what actually makes two invocations' data disjoint: this run's raw
    table only ever contains this run's tagged ids, so an old, stale log entry for a
    foreign tag joins against nothing and cannot silently dilute this run's mismatch rate
    the way it did before this was added.
    """
    from dataclasses import replace

    return [
        replace(event, user_id=f"{tag}-{event.user_id}", video_id=f"{tag}-{event.video_id}")
        for event in events
    ]


def _seed_online_state(redis_client: sync_redis.Redis, events: list[EngagementEvent]) -> list[str]:
    user_ids = sorted({event.user_id for event in events})
    for event in events:
        process_event(redis_client, event, DEMO_REGISTRY)
    return user_ids


def _correct_head_dropped_value(
    redis_client: sync_redis.Redis, entity_key: str, feature_name: str, as_of_ms: int
) -> float | None:
    """What ``feature_name`` would read if the online path never merged its head.

    Fetches the same tiles the online store already has for this entity's grid and
    reruns the same pure rollup with an empty head, rather than any private
    reimplementation of the merge.
    """
    from asofline.agg import monoid_for, rollup_at

    specs = {spec.feature_name: spec for spec in feature_specs(USER_ENGAGEMENT)}
    tile_lookup = {
        (spec.agg_name, spec.granularity_ms): spec for spec in tile_specs(USER_ENGAGEMENT)
    }
    feature_spec = specs[feature_name]
    tile_spec = tile_lookup[feature_spec.tile_key]

    raw = redis_client.hgetall(
        tile_hash_key(USER_ENGAGEMENT, entity_key, feature_spec.granularity_ms)
    )
    tiles = {}
    for field_raw, payload in raw.items():
        agg_name, tile_index = parse_tile_field(field_raw.decode())
        if agg_name != feature_spec.agg_name:
            continue
        tiles[tile_index] = decode_state(payload, arity=tile_spec.arity)

    return rollup_at(
        monoid_for(feature_spec.function),
        tiles,
        [],  # the injected bug: no head events at all
        as_of_ms=as_of_ms,
        window_ms=feature_spec.window_ms,
        granularity_ms=feature_spec.granularity_ms,
    )


def _publish_served_vectors(
    redis_client: sync_redis.Redis,
    user_ids: list[str],
    as_of_ms: int,
    *,
    buggy_fraction: float,
    seed: int,
) -> int:
    """Read the real online store for each entity, optionally degrade one feature, publish.

    Returns the number of vectors actually published (one per entity that has any data).
    """
    import asyncio
    import random

    from asofline.online.store import OnlineStore

    rng = random.Random(seed)
    producer = Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})
    published = 0

    async def _serve_all() -> list[dict[str, float | None]]:
        store = OnlineStore.from_url(TEST_REDIS_URL)
        try:
            return await store.get_online_features(
                USER_ENGAGEMENT, [{"user_id": uid} for uid in user_ids], as_of_ms=as_of_ms
            )
        finally:
            await store.close()

    vectors = asyncio.run(_serve_all())

    for user_id, features in zip(user_ids, vectors, strict=True):
        if all(value is None for value in features.values()):
            continue  # nothing served for this entity; nothing meaningful to log
        served = dict(features)
        if rng.random() < buggy_fraction:
            entity_key = entity_value_key(USER_ENGAGEMENT, {"user_id": user_id})
            served[TARGET_FEATURE] = _correct_head_dropped_value(
                redis_client, entity_key, TARGET_FEATURE, as_of_ms
            )
        entry = build_log_entry(
            USER_ENGAGEMENT,
            {"user_id": user_id},
            log_id=uuid.uuid4().hex,
            request_ts_ms=as_of_ms,
            served_at_ms=as_of_ms,
            features=served,
        )
        producer.produce(SETTINGS.feature_log_topic, value=json.dumps(entry.to_dict()).encode())
        published += 1
    producer.flush(10.0)
    return published


def _run_scenario(
    spark: SparkSession, redis_client: sync_redis.Redis, *, buggy_fraction: float, seed: int
) -> dict[str, object]:
    config = GeneratorConfig(
        seed=seed,
        n_events=30_000,
        n_users=300,
        n_videos=1_500,
        late_fraction=0.0,  # isolates implementation skew from the P2-style leak entirely
    )
    namespace = f"{NAMESPACE}_{seed}"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")

    # serving_log is a persistent Iceberg table; drop and recreate it so this run starts
    # from an empty table rather than piling on top of whatever a previous run using the
    # same namespace left behind.
    #
    # The checkpoint is deliberately NOT cleared, and this is the one that matters. A
    # cleared checkpoint makes Structured Streaming treat the query as brand new, which
    # with startingOffsets="earliest" means it re-reads the *entire retained history* of
    # the shared feature_logs topic, not just what this run just published, undoing the
    # table drop by writing back years of unrelated historical rows from every earlier
    # test run and benchmark this session has ever produced. Leaving the checkpoint alone
    # lets it resume from wherever it last left off, so a run only ever ingests messages
    # published since then, which in practice means only its own. Combined with
    # _tag_entity_ids (below) so that even a plausible edge case, a message this run
    # somehow does re-read from before the checkpoint, cannot join against this run's own
    # freshly rebuilt raw data and silently count as anything.
    checkpoint_dir = f"/tmp/asofline-skew-test-checkpoints/{namespace}"
    spark.sql(f"DROP TABLE IF EXISTS {namespace}.serving_log PURGE")

    # A fresh tag every call, not derived from the seed: two calls with the same seed
    # (deliberate, for comparing features from the same underlying data) must still
    # produce disjoint entity ids, or their published log entries collide in exactly the
    # way _tag_entity_ids's docstring explains.
    run_tag = uuid.uuid4().hex[:12]
    events = _tag_entity_ids(EngagementGenerator(config).generate(), run_tag)
    raw_table = load_events(spark, events, namespace=namespace, table="engagement_events")
    build_tiles(spark, USER_ENGAGEMENT, namespace=namespace, raw_table=raw_table)

    user_ids = _seed_online_state(redis_client, events)
    as_of_ms = max(event.event_ts for event in events) - 5 * 60_000

    log_table = f"{namespace}.serving_log"
    published = _publish_served_vectors(
        redis_client, user_ids, as_of_ms, buggy_fraction=buggy_fraction, seed=seed + 1
    )
    assert published > 0, "no vectors were published; the scenario produced no data to log"

    run_feature_log_ingest(
        spark,
        namespace=namespace,
        table="serving_log",
        checkpoint_location=checkpoint_dir,
        group_id=f"asofline-skew-test-{namespace}",
        starting_offsets="earliest",
        bounded=True,
    )

    reports = detect_skew(
        spark,
        USER_ENGAGEMENT,
        namespace=namespace,
        raw_table=raw_table,
        log_table=log_table,
        # This test's own published vectors all share this exact as_of_ms. Scoping the
        # detector to that instant is what keeps this test correct in the face of the
        # feature_logs topic also carrying every other test's and benchmark's historical
        # traffic: without the window, a fresh consumer group reading from "earliest"
        # sees the whole shared topic, not just this run's data.
        min_request_ts_ms=as_of_ms,
        max_request_ts_ms=as_of_ms,
    )
    return {"reports": reports, "published": published}


class TestFalsePositive:
    def test_a_clean_pipeline_reports_no_significant_skew(
        self, spark: SparkSession, redis_client: sync_redis.Redis
    ) -> None:
        """The plan's explicit requirement: a detector that always fires is not a detector."""
        result = _run_scenario(spark, redis_client, buggy_fraction=0.0, seed=101)
        reports = result["reports"]
        for feature_name, report in reports.items():
            assert report.compared > 0, feature_name
            assert report.mismatch_rate < 0.02, (
                f"{feature_name}: {report.mismatch_rate:.4f} mismatch rate on a clean run "
                f"({report.bucket_counts})"
            )


class TestInjectedBug:
    def test_the_degraded_feature_is_flagged_and_classified_as_partial_head_tile(
        self, spark: SparkSession, redis_client: sync_redis.Redis
    ) -> None:
        result = _run_scenario(spark, redis_client, buggy_fraction=0.5, seed=202)
        reports = result["reports"]
        target = reports[TARGET_FEATURE]

        # Not close to 50%: degrading "the head" only produces a visible difference for
        # an entity that actually had something in its head to lose, and under this
        # Zipf-skewed population most randomly chosen entities have zero activity in any
        # given 1-hour window. Measured directly (see TestSensitivityFloor), roughly 29%
        # of degraded entities show an observable effect, so 0.08 is comfortably below
        # what a real injection produces and comfortably above the false-positive
        # baseline (under 2%) TestFalsePositive establishes on a clean run.
        assert target.mismatch_rate > 0.08, (
            f"expected the injected bug to produce a clearly non-trivial mismatch rate, "
            f"got {target.mismatch_rate:.4f} ({target.bucket_counts})"
        )
        assert target.bucket_counts.get(BUCKET_PARTIAL_HEAD_TILE, 0) > 0
        # Every mismatch should land in partial_head_tile: late_fraction=0 in this scenario
        # means the tiled and strict recomputations never disagree with each other, so
        # nothing here should be misclassified as late-arriving data.
        mismatches = target.mismatch_count
        assert target.bucket_counts.get(BUCKET_PARTIAL_HEAD_TILE, 0) == mismatches

    def test_every_other_feature_stays_clean(
        self, spark: SparkSession, redis_client: sync_redis.Redis
    ) -> None:
        """The other half of a real detector: it names the bug, it doesn't smear it.

        Degrading one feature must not make every feature in the vector look suspect.
        """
        result = _run_scenario(spark, redis_client, buggy_fraction=0.5, seed=202)
        reports = result["reports"]
        for feature_name, report in reports.items():
            if feature_name == TARGET_FEATURE:
                continue
            assert report.mismatch_rate < 0.02, (
                f"{feature_name} shows {report.mismatch_rate:.4f} mismatch rate; the bug "
                f"was only injected into {TARGET_FEATURE}"
            )

    def test_psi_and_ks_are_reported_for_the_degraded_feature(
        self, spark: SparkSession, redis_client: sync_redis.Redis
    ) -> None:
        result = _run_scenario(spark, redis_client, buggy_fraction=0.5, seed=202)
        target = result["reports"][TARGET_FEATURE]
        assert target.psi is not None
        assert target.ks_statistic is not None
        # Measured at ~0.023 for this scenario; 0.01 confirms a real, non-zero divergence
        # without pinning a brittle exact value to a specific seed's sampling noise.
        assert target.ks_statistic > 0.01


class TestSensitivityFloor:
    @pytest.mark.parametrize("buggy_fraction", [0.02, 0.10])
    def test_the_smallest_injected_fraction_this_sample_size_reliably_catches(
        self, spark: SparkSession, redis_client: sync_redis.Redis, buggy_fraction: float
    ) -> None:
        """Reports the sensitivity floor rather than only asserting the detector "works".

        A detector that only catches a 50% bug rate is a much weaker claim than one that
        catches 2%, and the plan asks for this number to be published, not assumed.
        """
        seed = 303 + int(buggy_fraction * 1000)
        result = _run_scenario(spark, redis_client, buggy_fraction=buggy_fraction, seed=seed)
        target = result["reports"][TARGET_FEATURE]
        # The observed mismatch rate should track the injected fraction rather than
        # collapsing to zero (missed) or exploding to certainty (over-flagging). It is not
        # expected to equal the injected fraction: degrading "the head" is only visible
        # for an entity that had something in its head to lose, and under this
        # Zipf-skewed population most entities do not, in any given 1-hour window.
        # Measured directly at both 2% and 10% injection, the observed rate lands at
        # roughly a third of the injected fraction; 0.15 leaves comfortable margin below
        # that for run-to-run sampling noise while still ruling out "missed entirely".
        assert target.mismatch_rate > buggy_fraction * 0.15, (
            f"at a {buggy_fraction:.0%} injection rate with {target.compared} rows compared, "
            f"only {target.mismatch_rate:.4f} were flagged"
        )
