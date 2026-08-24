"""The resolution ladder, which both paths must agree on exactly."""

from __future__ import annotations

from datetime import timedelta

import pytest

from asofline.definitions import DefinitionError, Resolution
from asofline.definitions.resolution import FIVE_MINUTE_RESOLUTION, FIVE_MINUTES, ONE_HOUR

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
WEEK = timedelta(days=7)


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (timedelta(minutes=5), FIVE_MINUTES),
        (HOUR, FIVE_MINUTES),
        (timedelta(hours=12), FIVE_MINUTES),
        (timedelta(hours=13), ONE_HOUR),
        (DAY, ONE_HOUR),
        (WEEK, ONE_HOUR),
        (timedelta(days=21), ONE_HOUR),
    ],
)
def test_tier_assignment(window: timedelta, expected: timedelta) -> None:
    assert FIVE_MINUTE_RESOLUTION.granularity_for(window) == expected


def test_tile_counts_stay_under_the_cap() -> None:
    for window in (HOUR, DAY, WEEK, timedelta(days=21)):
        assert FIVE_MINUTE_RESOLUTION.tiles_in(window) <= 512


def test_windows_beyond_the_ladder_are_rejected_not_rounded() -> None:
    with pytest.raises(DefinitionError, match="exceeds the longest supported window"):
        FIVE_MINUTE_RESOLUTION.granularity_for(timedelta(days=22))


def test_window_that_is_not_a_whole_number_of_tiles_is_rejected() -> None:
    # 7 minutes is not a multiple of the 5-minute grid, so its trailing tile is partial
    # and has no owner. Rounding it silently is the failure this refuses.
    with pytest.raises(DefinitionError, match="not a whole multiple"):
        FIVE_MINUTE_RESOLUTION.granularity_for(timedelta(minutes=7))


def test_non_positive_window_is_rejected() -> None:
    with pytest.raises(DefinitionError, match="must be positive"):
        FIVE_MINUTE_RESOLUTION.granularity_for(timedelta(0))


def test_the_field_cap_is_enforced_not_merely_documented() -> None:
    """A single fine grid over a long window is what the tiering exists to prevent.

    This is the configuration the plan's original clamp rule would have produced for the
    demo view: a 5-minute grid answering 7 days, which needs 2016 fields.
    """
    single_fine_grid = Resolution(tiers=((timedelta(days=30), FIVE_MINUTES),))
    with pytest.raises(DefinitionError, match="above the cap"):
        single_fine_grid.granularity_for(WEEK)
    assert single_fine_grid.granularity_for(HOUR) == FIVE_MINUTES


def test_tiers_must_ascend() -> None:
    with pytest.raises(DefinitionError, match="strictly ascending"):
        Resolution(tiers=((WEEK, ONE_HOUR), (HOUR, FIVE_MINUTES)))


def test_granularities_are_reported_finest_first() -> None:
    assert FIVE_MINUTE_RESOLUTION.granularities == (FIVE_MINUTES, ONE_HOUR)
