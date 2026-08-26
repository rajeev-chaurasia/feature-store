"""``online.store.OnlineStore`` against a real Redis, seeded by hand.

The Kafka-to-Redis consumer that will normally write these keys is being built in
parallel and does not exist yet, so every fixture here is seeded directly through
``tests.redis_fixtures``, which writes in the exact shape ``online.store`` reads.
"""

from __future__ import annotations

import pytest

from asofline.agg.monoid import SUM
from asofline.agg.rollup import rollup_at
from asofline.demo.views import USER_ENGAGEMENT
from asofline.online.store import OnlineStore
from tests.redis_fixtures import (
    flush_test_database,
    make_test_redis,
    run_async,
    seed_head_event,
    seed_tile,
)

pytestmark = pytest.mark.integration

MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
FIVE_MIN = 5 * MINUTE

# 2026-01-01T00:12:00Z, an arbitrary fixed instant so every test is deterministic. Offset
# 12 minutes past midnight (rather than exactly on the hour) so it does not sit on every
# grid's tile boundary at once, which would make every grid's head window degenerate to
# empty and hide bugs in the head-merge path.
AS_OF_MS = 1_767_225_600_000 + 12 * 60_000


@run_async
async def test_single_entity_matches_rollup_computed_directly() -> None:
    redis = make_test_redis()
    await flush_test_database(redis)
    try:
        values = {"user_id": "u-single"}
        # watch_seconds_sum_1h lands on the 5-minute grid: two tiles inside the window,
        # plus one head event on the leading edge.
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            values,
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 20 * MINUTE,
            state=(10.0,),
        )
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            values,
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 10 * MINUTE,
            state=(5.0,),
        )
        # watch_seconds_sum_7d lands on the 1-hour grid.
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            values,
            agg_name="watch_seconds_sum",
            granularity_ms=HOUR,
            event_ts_ms=AS_OF_MS - 3 * HOUR,
            state=(100.0,),
        )
        await seed_head_event(
            redis,
            USER_ENGAGEMENT,
            values,
            event_ts_ms=AS_OF_MS - 30_000,
            columns={"watch_seconds": 2.5, "liked": 0.0},
        )

        store = OnlineStore(redis)
        try:
            [features] = await store.get_online_features(
                USER_ENGAGEMENT, [values], as_of_ms=AS_OF_MS
            )
        finally:
            await store.close()

        # Computed independently via the same shared rollup function, from the same raw
        # inputs, to prove the store did not reimplement window/merge logic on its own.
        # Both seeded 5-minute tiles (-20min, -10min) are well clear of the 5-minute
        # grid's head boundary (2 minutes before as_of here), so both are completed tiles
        # the window covers, plus the head event on the leading edge.
        tiles_5min = {
            (AS_OF_MS - 20 * MINUTE) // FIVE_MIN: (10.0,),
            (AS_OF_MS - 10 * MINUTE) // FIVE_MIN: (5.0,),
        }
        expected_1h = rollup_at(
            SUM,
            tiles_5min,
            [(AS_OF_MS - 30_000, 2.5)],
            as_of_ms=AS_OF_MS,
            window_ms=HOUR,
            granularity_ms=FIVE_MIN,
        )
        assert features["watch_seconds_sum_1h"] == pytest.approx(expected_1h)
        assert features["watch_seconds_sum_1h"] == pytest.approx(17.5)

        expected_7d = rollup_at(
            SUM,
            {AS_OF_MS // HOUR - 3: (100.0,)},
            [(AS_OF_MS - 30_000, 2.5)],
            as_of_ms=AS_OF_MS,
            window_ms=7 * DAY,
            granularity_ms=HOUR,
        )
        assert features["watch_seconds_sum_7d"] == pytest.approx(expected_7d)
        assert features["watch_seconds_sum_7d"] == pytest.approx(102.5)
    finally:
        await flush_test_database(redis)
        await redis.aclose()


@run_async
async def test_two_entities_in_one_call_both_come_back_correctly() -> None:
    redis = make_test_redis()
    await flush_test_database(redis)
    try:
        alice = {"user_id": "u-alice"}
        bob = {"user_id": "u-bob"}
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            alice,
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 10 * MINUTE,
            state=(7.0,),
        )
        await seed_head_event(
            redis, USER_ENGAGEMENT, alice, event_ts_ms=AS_OF_MS - 60_000, columns={}
        )
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            bob,
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 10 * MINUTE,
            state=(40.0,),
        )
        await seed_head_event(
            redis, USER_ENGAGEMENT, bob, event_ts_ms=AS_OF_MS - 60_000, columns={}
        )

        store = OnlineStore(redis)
        try:
            alice_features, bob_features = await store.get_online_features(
                USER_ENGAGEMENT, [alice, bob], as_of_ms=AS_OF_MS
            )
        finally:
            await store.close()

        assert alice_features["watch_seconds_sum_1h"] == pytest.approx(7.0)
        assert bob_features["watch_seconds_sum_1h"] == pytest.approx(40.0)
    finally:
        await flush_test_database(redis)
        await redis.aclose()


@run_async
async def test_an_entity_with_no_data_at_all_is_served_nulls() -> None:
    redis = make_test_redis()
    await flush_test_database(redis)
    try:
        store = OnlineStore(redis)
        try:
            [features] = await store.get_online_features(
                USER_ENGAGEMENT, [{"user_id": "u-never-seen"}], as_of_ms=AS_OF_MS
            )
        finally:
            await store.close()

        assert set(features) == set(USER_ENGAGEMENT.feature_names)
        assert all(value is None for value in features.values())
    finally:
        await flush_test_database(redis)
        await redis.aclose()


@run_async
async def test_an_entity_whose_only_data_is_older_than_ttl_is_served_nulls() -> None:
    redis = make_test_redis()
    await flush_test_database(redis)
    try:
        values = {"user_id": "u-stale"}
        # USER_ENGAGEMENT's ttl is 7 days; seed one tile well outside it.
        stale_event_ms = AS_OF_MS - 8 * 24 * HOUR
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            values,
            agg_name="watch_seconds_sum",
            granularity_ms=HOUR,
            event_ts_ms=stale_event_ms,
            state=(999.0,),
        )

        store = OnlineStore(redis)
        try:
            [features] = await store.get_online_features(
                USER_ENGAGEMENT, [values], as_of_ms=AS_OF_MS
            )
        finally:
            await store.close()

        assert all(value is None for value in features.values())
    finally:
        await flush_test_database(redis)
        await redis.aclose()


@run_async
async def test_multi_entity_request_issues_exactly_one_pipeline_round_trip() -> None:
    """The perf requirement: N entities in one request, one ``pipeline.execute()``."""
    redis = make_test_redis()
    await flush_test_database(redis)
    try:
        values = [{"user_id": f"u-perf-{i}"} for i in range(5)]
        for entity in values:
            await seed_tile(
                redis,
                USER_ENGAGEMENT,
                entity,
                agg_name="watch_seconds_sum",
                granularity_ms=FIVE_MIN,
                event_ts_ms=AS_OF_MS - 10 * MINUTE,
                state=(1.0,),
            )

        store = OnlineStore(redis)
        execute_calls = 0
        original_pipeline = redis.pipeline

        def counting_pipeline(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            pipe = original_pipeline(*args, **kwargs)
            original_execute = pipe.execute

            async def counting_execute(*e_args: object, **e_kwargs: object):  # type: ignore[no-untyped-def]
                nonlocal execute_calls
                execute_calls += 1
                return await original_execute(*e_args, **e_kwargs)

            pipe.execute = counting_execute  # type: ignore[method-assign]
            return pipe

        redis.pipeline = counting_pipeline  # type: ignore[method-assign]
        try:
            results = await store.get_online_features(USER_ENGAGEMENT, values, as_of_ms=AS_OF_MS)
        finally:
            await store.close()

        assert execute_calls == 1
        assert len(results) == 5
        assert all(r["watch_seconds_sum_1h"] == pytest.approx(1.0) for r in results)
    finally:
        await flush_test_database(redis)
        await redis.aclose()
