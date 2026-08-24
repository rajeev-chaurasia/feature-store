"""Window bounds. Integer arithmetic, and the exact place the two paths must agree."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from asofline.agg import align_down, bounds_for, retention_start_index, tile_index

MINUTE = 60_000
FIVE_MIN = 5 * MINUTE
HOUR = 60 * MINUTE
DAY = 24 * HOUR

timestamps = st.integers(min_value=0, max_value=4_000_000_000_000)
granularities = st.sampled_from([MINUTE, FIVE_MIN, HOUR])
multipliers = st.integers(min_value=1, max_value=500)


class TestAlignment:
    def test_align_down_is_idempotent(self) -> None:
        assert align_down(align_down(1_234_567, FIVE_MIN), FIVE_MIN) == align_down(
            1_234_567, FIVE_MIN
        )

    def test_a_boundary_aligns_to_itself(self) -> None:
        assert align_down(2 * FIVE_MIN, FIVE_MIN) == 2 * FIVE_MIN

    def test_tile_index_uses_floor_not_truncation(self) -> None:
        """Truncating division folds the two tiles either side of the epoch into one."""
        assert tile_index(-1, FIVE_MIN) == -1
        assert tile_index(0, FIVE_MIN) == 0
        assert tile_index(-FIVE_MIN, FIVE_MIN) == -1


class TestBounds:
    def test_the_head_is_empty_exactly_on_a_boundary(self) -> None:
        aligned = bounds_for(as_of_ms=100 * HOUR, window_ms=DAY, granularity_ms=HOUR)
        assert aligned.head_is_empty
        offset = bounds_for(as_of_ms=100 * HOUR + 1, window_ms=DAY, granularity_ms=HOUR)
        assert not offset.head_is_empty
        assert offset.head_start_ms == 100 * HOUR

    def test_effective_window_is_between_w_and_w_plus_g(self) -> None:
        @given(as_of=timestamps, granularity=granularities, multiple=multipliers)
        def check(as_of: int, granularity: int, multiple: int) -> None:
            window = granularity * multiple
            bounds = bounds_for(as_of, window, granularity)
            assert window <= bounds.effective_window_ms < window + granularity

        check()

    def test_tiles_and_head_tile_the_effective_window_exactly(self) -> None:
        """No gap and no overlap between the whole tiles and the exact head."""

        @given(as_of=timestamps, granularity=granularities, multiple=multipliers)
        def check(as_of: int, granularity: int, multiple: int) -> None:
            bounds = bounds_for(as_of, granularity * multiple, granularity)
            assert bounds.effective_start_ms == bounds.tile_start_index * granularity
            assert bounds.tile_start_index + bounds.tile_count == bounds.tile_end_index
            assert bounds.tile_end_index * granularity == bounds.head_start_ms
            assert bounds.head_start_ms <= bounds.as_of_ms

        check()

    def test_as_of_is_exclusive_and_the_effective_start_is_inclusive(self) -> None:
        bounds = bounds_for(as_of_ms=10 * HOUR + 90_000, window_ms=2 * HOUR, granularity_ms=HOUR)
        assert not bounds.covers_head_event(bounds.as_of_ms)
        assert bounds.covers_head_event(bounds.as_of_ms - 1)
        assert bounds.covers_head_event(bounds.head_start_ms)
        assert not bounds.covers_head_event(bounds.head_start_ms - 1)
        assert bounds.covers_tile(bounds.tile_start_index)
        assert not bounds.covers_tile(bounds.tile_start_index - 1)
        assert not bounds.covers_tile(bounds.tile_end_index)

    def test_a_window_of_one_tile_ending_on_a_boundary_has_one_tile_and_no_head(self) -> None:
        bounds = bounds_for(as_of_ms=10 * HOUR, window_ms=HOUR, granularity_ms=HOUR)
        assert bounds.tile_count == 1
        assert bounds.head_is_empty

    def test_a_window_of_one_tile_mid_tile_has_one_tile_and_a_head(self) -> None:
        """The effective window is longer than the nominal one. That is the snap."""
        bounds = bounds_for(as_of_ms=10 * HOUR + 1, window_ms=HOUR, granularity_ms=HOUR)
        assert bounds.tile_count == 1
        assert not bounds.head_is_empty
        assert bounds.effective_window_ms == HOUR + 1


class TestValidation:
    def test_window_not_a_multiple_of_granularity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a whole multiple"):
            bounds_for(as_of_ms=0, window_ms=7 * MINUTE, granularity_ms=FIVE_MIN)

    @pytest.mark.parametrize(("window", "granularity"), [(0, HOUR), (HOUR, 0), (-HOUR, HOUR)])
    def test_non_positive_arguments_are_rejected(self, window: int, granularity: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            bounds_for(as_of_ms=0, window_ms=window, granularity_ms=granularity)


class TestRetention:
    def test_retention_start_is_the_oldest_readable_tile(self) -> None:
        as_of = 100 * HOUR
        start = retention_start_index(as_of, retention_ms=DAY, granularity_ms=HOUR)
        # The longest window on this grid, evaluated now, cannot reach below this index.
        bounds = bounds_for(as_of, window_ms=DAY, granularity_ms=HOUR)
        assert start == bounds.tile_start_index
