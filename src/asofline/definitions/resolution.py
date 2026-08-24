"""How a window length maps to a tile granularity.

This is the single most load-bearing policy in the project, because both the batch path
and the streaming path consult it and must agree on it exactly. If they disagree, every
served value is wrong in a way that looks like noise.

The plan originally specified one granularity per view, ``clamp(smallest/12, 60s,
longest/512)``. That rule is wrong, and wrong in a way worth recording. It uses
``longest/512`` as an upper bound on ``g``, but the field cap makes it a *lower* bound:
to keep the longest window under 512 tiles you need ``g >= longest/512``. The two real
constraints are

    g <= smallest / 12      (trailing snap error stays small on the shortest window)
    g >= longest  / 512     (the longest window fits in one hash read)

which are simultaneously satisfiable only when ``longest / smallest <= 512/12``, about
42. The plan's own example view declares 1h, 24h and 7d, a ratio of 168, so no single
granularity satisfies it. Evaluating the clamp on that view returns 5 minutes and the 7d
window then needs 2016 fields, four times the cap, silently.

The fix is the design Chronon actually ships: tiered resolution. Short windows land on a
fine grid, long windows on a coarse one, and each window is answered from exactly one
grid. Tile counts stay bounded on both, and the trailing snap error stays a small
fraction of the window it applies to:

    1h  window on a 5-minute grid ->  12 tiles, snap error <= 5 minutes  (8% of 1h)
    24h window on a 1-hour   grid ->  24 tiles, snap error <= 1 hour     (4% of 24h)
    7d  window on a 1-hour   grid -> 168 tiles, snap error <= 1 hour     (0.6% of 7d)

The 12-hour tier bound is what puts 24h on the coarse grid rather than the fine one. A
5-minute grid would answer 24h in 288 fields, inside the cap, and the tier bound could be
raised to a day to get it. It is not, because the fine grid then has to be retained for a
day rather than an hour, which is a 24x increase in online tile storage for a snap error
improvement nobody asked for. The tier bound is the knob if that trade ever changes.

The cost is that an event updates one tile per grid it touches rather than one tile
total. That is two Redis field updates in the common case, inside one pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from asofline.definitions.errors import DefinitionError

FIVE_MINUTES = timedelta(minutes=5)
ONE_HOUR = timedelta(hours=1)
TWELVE_HOURS = timedelta(hours=12)
TWENTY_ONE_DAYS = timedelta(days=21)


@dataclass(frozen=True, slots=True)
class Resolution:
    """A ladder of ``(max_window, granularity)`` tiers, consulted shortest tier first.

    ``max_tiles_per_window`` bounds how many hash fields one window can require, which is
    what keeps a single online read to one round trip per grid.
    """

    tiers: tuple[tuple[timedelta, timedelta], ...]
    max_tiles_per_window: int = 512

    def __post_init__(self) -> None:
        if not self.tiers:
            raise DefinitionError("a resolution needs at least one tier")
        bounds = [bound for bound, _ in self.tiers]
        if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise DefinitionError(f"resolution tiers must be strictly ascending, got {bounds}")
        for bound, granularity in self.tiers:
            if granularity <= timedelta(0):
                raise DefinitionError(f"tier granularity must be positive, got {granularity!r}")
            if bound <= timedelta(0):
                raise DefinitionError(f"tier bound must be positive, got {bound!r}")

    @property
    def longest_supported_window(self) -> timedelta:
        return self.tiers[-1][0]

    @property
    def granularities(self) -> tuple[timedelta, ...]:
        """Every distinct grid this resolution can place a tile on, finest first."""
        seen: list[timedelta] = []
        for _, granularity in self.tiers:
            if granularity not in seen:
                seen.append(granularity)
        return tuple(sorted(seen))

    def granularity_for(self, window: timedelta) -> timedelta:
        """The grid that answers ``window``, or raise if no tier can.

        Rejecting is deliberate. Silently rounding a window the ladder cannot express is
        the exact failure this class exists to prevent.
        """
        if window <= timedelta(0):
            raise DefinitionError(f"window must be positive, got {window!r}")
        for bound, granularity in self.tiers:
            if window <= bound:
                self._check(window, granularity)
                return granularity
        raise DefinitionError(
            f"window {window!r} exceeds the longest supported window "
            f"{self.longest_supported_window!r}; add a coarser tier or shorten the window"
        )

    def tiles_in(self, window: timedelta) -> int:
        """How many whole tiles the trailing part of ``window`` spans."""
        granularity = self.granularity_for(window)
        return int(window.total_seconds() // granularity.total_seconds())

    def _check(self, window: timedelta, granularity: timedelta) -> None:
        window_seconds = window.total_seconds()
        granularity_seconds = granularity.total_seconds()
        if window_seconds % granularity_seconds != 0:
            raise DefinitionError(
                f"window {window!r} is not a whole multiple of its tile granularity "
                f"{granularity!r}; a partial trailing tile has no well defined owner"
            )
        tiles = int(window_seconds // granularity_seconds)
        if tiles > self.max_tiles_per_window:
            raise DefinitionError(
                f"window {window!r} needs {tiles} tiles at granularity {granularity!r}, "
                f"above the cap of {self.max_tiles_per_window}"
            )


FIVE_MINUTE_RESOLUTION = Resolution(
    tiers=(
        (TWELVE_HOURS, FIVE_MINUTES),
        (TWENTY_ONE_DAYS, ONE_HOUR),
    )
)
"""The default ladder. Windows up to 12 hours resolve to 5 minutes, up to 21 days to 1 hour."""
