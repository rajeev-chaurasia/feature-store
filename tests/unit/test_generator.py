"""The generator underwrites every committed number, so its properties are tested.

If the stream is not reproducible, the artifact validator is checking arithmetic on sand.
If the stream has no late tail, a leaky backfill passes every point-in-time test.
"""

from __future__ import annotations

import collections
from datetime import UTC, datetime, timedelta

import pytest

from asofline.demo.events import EngagementEvent, EventType, from_millis, to_millis
from asofline.demo.generator import EngagementGenerator, GeneratorConfig

SMALL = GeneratorConfig(n_events=20_000, n_users=500, n_videos=2_000)


@pytest.fixture(scope="module")
def events() -> list[EngagementEvent]:
    return EngagementGenerator(SMALL).generate()


class TestDeterminism:
    def test_same_seed_gives_a_byte_identical_stream(self) -> None:
        first = EngagementGenerator(SMALL).generate()
        second = EngagementGenerator(SMALL).generate()
        assert first == second

    def test_a_different_seed_gives_a_different_stream(self) -> None:
        from dataclasses import replace

        other = EngagementGenerator(replace(SMALL, seed=SMALL.seed + 1)).generate()
        assert other != EngagementGenerator(SMALL).generate()

    def test_config_is_hashable_so_an_artifact_can_name_its_stream(self) -> None:
        assert hash(SMALL) == hash(GeneratorConfig(n_events=20_000, n_users=500, n_videos=2_000))


class TestLateArrival:
    def test_creation_never_precedes_the_event(self, events: list[EngagementEvent]) -> None:
        assert all(event.created_ts >= event.event_ts for event in events)

    def test_an_event_created_before_it_happened_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="before it happened"):
            EngagementEvent(
                event_id="e0",
                event_type=EventType.WATCH,
                user_id="u0",
                video_id="v0",
                event_ts=1_000,
                created_ts=900,
            )

    def test_the_late_tail_is_present_and_close_to_the_configured_fraction(
        self, events: list[EngagementEvent]
    ) -> None:
        late = sum(1 for event in events if event.lateness_ms > SMALL.prompt_delay_ms)
        assert late / len(events) == pytest.approx(SMALL.late_fraction, abs=0.01)

    def test_the_tail_is_heavy_not_merely_present(self, events: list[EngagementEvent]) -> None:
        """A uniform delay would satisfy the fraction test and teach nothing.

        The point of the tail is that some events arrive long after the entity timestamps
        they should not have influenced, so the p99 has to be minutes, not milliseconds.
        """
        lateness = sorted(event.lateness_ms for event in events)
        p99 = lateness[int(0.99 * len(lateness))]
        assert p99 > 30_000, f"p99 lateness is only {p99}ms"

    def test_lateness_is_bounded(self, events: list[EngagementEvent]) -> None:
        assert max(event.lateness_ms for event in events) <= 6 * 60 * 60 * 1000


class TestOrdering:
    def test_events_arrive_in_creation_order(self, events: list[EngagementEvent]) -> None:
        keys = [(event.created_ts, event.event_id) for event in events]
        assert keys == sorted(keys)

    def test_event_time_is_not_sorted(self, events: list[EngagementEvent]) -> None:
        """If it were, the late tail would have been sorted away."""
        event_times = [event.event_ts for event in events]
        assert event_times != sorted(event_times)


class TestSkew:
    @pytest.mark.parametrize(("key", "population"), [("user_id", 500), ("video_id", 2_000)])
    def test_the_head_is_far_above_uniform(
        self, events: list[EngagementEvent], key: str, population: int
    ) -> None:
        counts = collections.Counter(getattr(event, key) for event in events)
        hottest = counts.most_common(1)[0][1] / len(events)
        assert hottest > 5 / population, "distribution is close to uniform, so no hot key"

    def test_videos_are_more_skewed_than_users(self, events: list[EngagementEvent]) -> None:
        """A single viewer generating a sixth of all traffic is not realistic.

        A single video doing so is. The two exponents differ for that reason, and this
        pins the ordering so a future tweak cannot silently invert it.
        """
        users = collections.Counter(event.user_id for event in events)
        videos = collections.Counter(event.video_id for event in events)
        assert videos.most_common(1)[0][1] > users.most_common(1)[0][1]


class TestPayload:
    def test_watch_seconds_only_on_watches(self, events: list[EngagementEvent]) -> None:
        for event in events:
            if event.event_type is not EventType.WATCH:
                assert event.watch_seconds == 0.0

    def test_flags_match_their_event_type(self, events: list[EngagementEvent]) -> None:
        for event in events:
            assert event.liked == (1 if event.event_type is EventType.LIKE else 0)
            assert event.shared == (1 if event.event_type is EventType.SHARE else 0)

    def test_round_trip_through_the_wire_form(self, events: list[EngagementEvent]) -> None:
        for event in events[:200]:
            assert EngagementEvent.from_dict(event.to_dict()) == event

    def test_ids_are_unique(self, events: list[EngagementEvent]) -> None:
        assert len({event.event_id for event in events}) == len(events)


class TestConfigValidation:
    def test_naive_start_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone aware"):
            GeneratorConfig(start=datetime(2026, 8, 1))

    def test_infinite_mean_tail_is_rejected(self) -> None:
        # With alpha <= 1 the Pareto has no finite mean, so the observed late fraction is
        # decided by whichever single draw happened to be largest.
        with pytest.raises(ValueError, match="late_delay_alpha must exceed 1"):
            GeneratorConfig(late_delay_alpha=1.0)

    def test_out_of_range_late_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="late_fraction"):
            GeneratorConfig(late_fraction=1.5)

    @pytest.mark.parametrize("field", ["n_users", "n_videos", "n_events"])
    def test_empty_populations_are_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            GeneratorConfig(**{field: 0})  # type: ignore[arg-type]


class TestTimeHelpers:
    def test_millis_round_trip(self) -> None:
        moment = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)
        assert from_millis(to_millis(moment)) == moment

    def test_naive_datetime_is_refused(self) -> None:
        with pytest.raises(ValueError, match="naive datetime"):
            to_millis(datetime(2026, 8, 23))

    def test_events_span_the_configured_duration(self, events: list[EngagementEvent]) -> None:
        span = max(e.event_ts for e in events) - min(e.event_ts for e in events)
        assert timedelta(milliseconds=span) <= SMALL.duration
        assert timedelta(milliseconds=span) > SMALL.duration * 0.99
