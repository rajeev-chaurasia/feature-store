"""Rollup: is tiling a faithful refactoring of the definition, or a different definition?

The differential test in ``TestTilingIsFaithful`` is the one that matters. Everything the
batch path and the online path do to a window is supposed to be an optimisation of
"filter the raw events and fold them". If it is not, both paths are consistently wrong
together and the skew detector will report nothing.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from asofline.agg import (
    AVG,
    COUNT,
    MAX,
    MIN,
    SUM,
    Monoid,
    bounds_for,
    brute_force,
    rollup,
    rollup_at,
    tiles_from_events,
)

MINUTE = 60_000
FIVE_MIN = 5 * MINUTE
HOUR = 60 * MINUTE
DAY = 24 * HOUR

ALL = [SUM, COUNT, MIN, MAX, AVG]

# A four-hour span at millisecond resolution, so events land mid-tile far more often
# than on a boundary.
event_streams = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=4 * HOUR),
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    ),
    min_size=0,
    max_size=60,
)
as_ofs = st.integers(min_value=0, max_value=4 * HOUR)


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return left == pytest.approx(right, rel=1e-9, abs=1e-9)


class TestTilingIsFaithful:
    """rollup(tiles(events)) must equal brute_force(events) for every function."""

    @pytest.mark.parametrize("monoid", ALL, ids=lambda m: str(m.function))
    @pytest.mark.parametrize("granularity", [MINUTE, FIVE_MIN, HOUR])
    def test_differential_against_the_oracle(self, monoid: Monoid, granularity: int) -> None:
        @settings(max_examples=150)
        @given(events=event_streams, as_of=as_ofs)
        def check(events: list[tuple[int, float]], as_of: int) -> None:
            window = granularity * 4
            bounds = bounds_for(as_of, window, granularity)
            tiled = rollup(monoid, tiles_from_events(monoid, events, granularity), events, bounds)
            oracle = brute_force(
                monoid,
                events,
                as_of_ms=as_of,
                window_ms=window,
                granularity_ms=granularity,
            )
            assert _same(tiled, oracle), f"{tiled!r} != {oracle!r}"

        check()

    @pytest.mark.parametrize("monoid", ALL, ids=lambda m: str(m.function))
    def test_holds_when_the_window_spans_many_tiles(self, monoid: Monoid) -> None:
        @settings(max_examples=80)
        @given(events=event_streams, as_of=as_ofs)
        def check(events: list[tuple[int, float]], as_of: int) -> None:
            tiled = rollup_at(
                monoid,
                tiles_from_events(monoid, events, FIVE_MIN),
                events,
                as_of_ms=as_of,
                window_ms=DAY,
                granularity_ms=FIVE_MIN,
            )
            oracle = brute_force(
                monoid, events, as_of_ms=as_of, window_ms=DAY, granularity_ms=FIVE_MIN
            )
            assert _same(tiled, oracle)

        check()


class TestNoDoubleCounting:
    """The head tile is served from raw events, so it must not also come from a tile.

    Callers are allowed to pass the whole event stream as ``head_events``; rollup filters
    it. If the filter were dropped, every event in the current tile would be counted
    twice, which is a small error that grows and shrinks with position inside the tile.
    """

    def test_an_event_in_the_head_tile_is_counted_once(self) -> None:
        events = [(90 * MINUTE, 5.0)]
        as_of = 95 * MINUTE
        tiles = tiles_from_events(SUM, events, HOUR)
        assert rollup_at(
            SUM, tiles, events, as_of_ms=as_of, window_ms=2 * HOUR, granularity_ms=HOUR
        ) == pytest.approx(5.0)

    def test_tiles_outside_the_window_are_ignored(self) -> None:
        events = [(0, 1.0), (10 * HOUR, 2.0)]
        tiles = tiles_from_events(SUM, events, HOUR)
        # A 2h window at t=10h30m reaches back to 8h, so the event at t=0 is excluded
        # even though its tile is present in the mapping.
        assert rollup_at(
            SUM,
            tiles,
            events,
            as_of_ms=10 * HOUR + 30 * MINUTE,
            window_ms=2 * HOUR,
            granularity_ms=HOUR,
        ) == pytest.approx(2.0)


class TestBoundarySemantics:
    def test_an_event_at_exactly_as_of_is_excluded(self) -> None:
        """The smallest possible label leak, and the hardest to notice."""
        events = [(HOUR, 1.0)]
        assert (
            rollup_at(
                COUNT,
                tiles_from_events(COUNT, events, FIVE_MIN),
                events,
                as_of_ms=HOUR,
                window_ms=HOUR,
                granularity_ms=FIVE_MIN,
            )
            == 0.0
        )

    def test_an_event_one_millisecond_earlier_is_included(self) -> None:
        events = [(HOUR - 1, 1.0)]
        assert (
            rollup_at(
                COUNT,
                tiles_from_events(COUNT, events, FIVE_MIN),
                events,
                as_of_ms=HOUR,
                window_ms=HOUR,
                granularity_ms=FIVE_MIN,
            )
            == 1.0
        )

    def test_the_trailing_edge_snaps_outward_not_inward(self) -> None:
        """An event just outside the nominal window is still counted, because the
        trailing edge snapped back to a tile boundary. The effective window is longer
        than the nominal one, never shorter, and this pins that direction."""
        as_of = 10 * HOUR + 30 * MINUTE
        just_outside = as_of - HOUR - MINUTE  # one minute before the nominal 1h window
        events = [(just_outside, 1.0)]
        bounds = bounds_for(as_of, HOUR, HOUR)
        assert bounds.effective_start_ms < as_of - HOUR
        assert (
            rollup_at(
                COUNT,
                tiles_from_events(COUNT, events, HOUR),
                events,
                as_of_ms=as_of,
                window_ms=HOUR,
                granularity_ms=HOUR,
            )
            == 1.0
        )


class TestEmptyAndAbsent:
    @pytest.mark.parametrize(
        ("monoid", "expected"),
        [(SUM, 0.0), (COUNT, 0.0), (MIN, None), (MAX, None), (AVG, None)],
    )
    def test_an_empty_window_finalizes_from_the_identity(
        self, monoid: Monoid, expected: float | None
    ) -> None:
        """Zero for the summable functions, null for the ones with nothing to report.

        This falls out of the identity rather than a special case, so the batch path and
        the online path cannot disagree about what "no events" means.
        """
        result = rollup_at(monoid, {}, [], as_of_ms=HOUR, window_ms=HOUR, granularity_ms=FIVE_MIN)
        assert _same(result, expected)

    def test_a_window_that_misses_every_event_is_also_empty(self) -> None:
        """Distinct from having no data at all, and deliberately not distinguished here.

        Whether an entity with no recent activity should read 0 or null is a serving
        concern, decided against the entity's last-seen marker and its ttl. Rollup
        answers only the arithmetic question.
        """
        events = [(0, 5.0)]
        result = rollup_at(
            SUM,
            tiles_from_events(SUM, events, FIVE_MIN),
            events,
            as_of_ms=10 * HOUR,
            window_ms=HOUR,
            granularity_ms=FIVE_MIN,
        )
        assert result == 0.0


class TestMergeOrderIsCanonical:
    def test_tile_iteration_order_does_not_change_the_answer(self) -> None:
        """A dict from Redis and a dict from Spark will not have the same key order.

        rollup sorts, so the result is identical. Without the sort the two paths would
        disagree in the last bits and the detector would report it every single run.
        """
        events = [(i * FIVE_MIN + 1, 0.1 * (i + 1)) for i in range(12)]
        tiles = tiles_from_events(SUM, events, FIVE_MIN)
        shuffled = dict(reversed(list(tiles.items())))
        bounds = bounds_for(HOUR, HOUR, FIVE_MIN)
        assert rollup(SUM, tiles, [], bounds) == rollup(SUM, shuffled, [], bounds)

    def test_head_event_order_does_not_change_the_answer(self) -> None:
        head = [(HOUR + i, 0.1 * (i + 1)) for i in range(20)]
        bounds = bounds_for(HOUR + 100, HOUR, FIVE_MIN)
        assert rollup(SUM, {}, head, bounds) == rollup(SUM, {}, list(reversed(head)), bounds)
