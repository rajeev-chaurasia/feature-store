"""The load generator's scheduling and statistics-collection logic, tested with no server.

``httpx.MockTransport`` stands in for a real serving process. What is under test here is
never "does the serving API work" (that belongs to tests/serving/), only: does the
generator schedule requests at the right cadence, does it measure latency from the
intended send time rather than the actual one, does it survive a failing request, and does
it hand back every completed sample.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from asofline.bench.load import (
    LoadTestConfig,
    intended_send_times,
    run_load_test,
    sample_entity_pool,
)


class TestSchedule:
    def test_request_count_matches_qps_times_duration(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=10,
            duration_s=2.0,
            entity_pool_size=5,
        )
        assert config.request_count == 20
        assert len(intended_send_times(config)) == 20

    def test_schedule_is_evenly_spaced(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=4,
            duration_s=1.0,
            entity_pool_size=5,
        )
        schedule = intended_send_times(config)
        from itertools import pairwise

        gaps = {round(b - a, 9) for a, b in pairwise(schedule)}
        assert gaps == {0.25}

    def test_schedule_is_reproducible_with_no_clock(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=7,
            duration_s=3.0,
            entity_pool_size=5,
        )
        assert intended_send_times(config) == intended_send_times(config)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [("qps", 0), ("qps", -1), ("duration_s", 0), ("entity_pool_size", 0)],
    )
    def test_invalid_config_is_rejected(self, field_name: str, value: float) -> None:
        kwargs = {
            "base_url": "http://x",
            "view": "user_engagement",
            "qps": 5,
            "duration_s": 1.0,
            "entity_pool_size": 5,
        }
        kwargs[field_name] = value
        with pytest.raises(ValueError, match=field_name):
            LoadTestConfig(**kwargs)  # type: ignore[arg-type]


class TestEntityPool:
    def test_pool_is_deterministic(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=5,
            duration_s=1.0,
            entity_pool_size=50,
            seed=7,
        )
        assert sample_entity_pool(config, "user_id") == sample_entity_pool(config, "user_id")

    def test_pool_has_the_configured_size(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=5,
            duration_s=1.0,
            entity_pool_size=17,
            seed=7,
        )
        assert len(sample_entity_pool(config, "user_id")) == 17


def _echo_transport(*, delay_ms: float = 0.0, fail_every: int | None = None) -> httpx.MockTransport:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if fail_every and calls["n"] % fail_every == 0:
            return httpx.Response(500, json={"error": "synthetic failure"})
        return httpx.Response(200, json={"as_of_ms": 0, "results": []})

    return httpx.MockTransport(handler)


class TestRunLoadTest:
    def test_every_request_produces_one_sample_on_success(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=20,
            duration_s=0.5,
            entity_pool_size=10,
        )

        async def go() -> None:
            async with httpx.AsyncClient(
                transport=_echo_transport(), base_url=config.base_url
            ) as client:
                result = await run_load_test(client, config, "user_id")
            assert result.completed_count == config.request_count
            assert result.error_count == 0
            assert all(sample >= 0.0 for sample in result.raw_samples_ms)

        asyncio.run(go())

    def test_failures_are_counted_and_do_not_produce_a_latency_sample(self) -> None:
        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=20,
            duration_s=0.5,
            entity_pool_size=10,
        )

        async def go() -> None:
            async with httpx.AsyncClient(
                transport=_echo_transport(fail_every=3), base_url=config.base_url
            ) as client:
                result = await run_load_test(client, config, "user_id")
            assert result.error_count > 0
            assert result.completed_count + result.error_count == result.requested_count

        asyncio.run(go())

    def test_latency_is_measured_from_intended_not_actual_send_time(self) -> None:
        """The whole point of the open-loop design, pinned down with an artificial stall.

        The first request's handler sleeps, which delays everything scheduled after it if
        the client were closed-loop. Because dispatch is not gated on completion, later
        requests still go out near their intended time and their measured latency reflects
        their own (near-zero) service time, not the earlier request's stall bleeding into
        their queueing delay from a serialized client.
        """
        state = {"first": True}

        async def handler(request: httpx.Request) -> httpx.Response:
            # An async handler, not a synchronous one with time.sleep: MockTransport
            # calls the handler in-process, so a blocking sleep there would freeze the
            # whole event loop and every other pending task with it, which is not what a
            # slow *server* does and would invalidate exactly the thing under test.
            if state["first"]:
                state["first"] = False
                await asyncio.sleep(0.2)
            return httpx.Response(200, json={"as_of_ms": 0, "results": []})

        config = LoadTestConfig(
            base_url="http://x",
            view="user_engagement",
            qps=20,
            duration_s=0.3,
            entity_pool_size=10,
        )

        async def go() -> None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url=config.base_url
            ) as client:
                result = await run_load_test(client, config, "user_id")
            # Everything after the stalled first request should still complete quickly
            # relative to intended time, since dispatch was never blocked on it.
            assert result.completed_count >= config.request_count - 1
            later = sorted(result.raw_samples_ms)[:-1]
            assert max(later) < 150, f"a later request paid for the earlier stall: {later}"

        asyncio.run(go())
