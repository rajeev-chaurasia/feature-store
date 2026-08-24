"""Window arithmetic: which tiles a window covers, and where the exact head begins.

This module is small and entirely integer arithmetic, and it is the thing both paths must
agree on to the millisecond. Everything about tiling that could be argued over is decided
here, once:

* **``as_of`` is exclusive.** An event at exactly ``T`` is not in the window ending at
  ``T``. Choosing the inclusive convention would let a training row see the very event it
  is trying to predict, which is the smallest possible instance of label leakage and the
  hardest to notice.
* **The leading edge is exact, the trailing edge snaps.** Whole tiles cover
  ``[align_down(T - W, g), align_down(T, g))`` and the head ``[align_down(T, g), T)`` is
  computed from raw events. So a freshly ingested event is visible immediately, and the
  effective window length is in ``[W, W + g)`` rather than exactly ``W``.

The asymmetry is deliberate. Snapping the leading edge too would make the two paths agree
just as well while costing up to ``g`` of freshness, which is the number P4 exists to
measure. Making the trailing edge exact instead would require raw events going back a
full window, which is the storage the tiles exist to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass


def align_down(timestamp_ms: int, granularity_ms: int) -> int:
    """The start of the tile containing ``timestamp_ms``."""
    return (timestamp_ms // granularity_ms) * granularity_ms


def tile_index(timestamp_ms: int, granularity_ms: int) -> int:
    """The index of the tile containing ``timestamp_ms``.

    Floor division, so this stays correct for timestamps before the epoch. Truncating
    division would fold the two tiles either side of zero into one.
    """
    return timestamp_ms // granularity_ms


@dataclass(frozen=True, slots=True)
class WindowBounds:
    """Everything a rollup needs to know about one ``(window, as_of)`` pair."""

    as_of_ms: int
    window_ms: int
    granularity_ms: int
    tile_start_index: int
    """Inclusive."""
    tile_end_index: int
    """Exclusive. Equal to ``tile_start_index`` when the window is shorter than one tile."""
    head_start_ms: int
    """Inclusive. The head is ``[head_start_ms, as_of_ms)`` and is read from raw events."""

    @property
    def effective_start_ms(self) -> int:
        return self.tile_start_index * self.granularity_ms

    @property
    def effective_window_ms(self) -> int:
        """In ``[window_ms, window_ms + granularity_ms)`` by construction."""
        return self.as_of_ms - self.effective_start_ms

    @property
    def tile_count(self) -> int:
        return self.tile_end_index - self.tile_start_index

    @property
    def head_is_empty(self) -> bool:
        """True exactly when ``as_of`` falls on a tile boundary."""
        return self.head_start_ms == self.as_of_ms

    def covers_tile(self, index: int) -> bool:
        return self.tile_start_index <= index < self.tile_end_index

    def covers_head_event(self, event_ts_ms: int) -> bool:
        return self.head_start_ms <= event_ts_ms < self.as_of_ms


def bounds_for(as_of_ms: int, window_ms: int, granularity_ms: int) -> WindowBounds:
    if granularity_ms <= 0:
        raise ValueError(f"granularity must be positive, got {granularity_ms}")
    if window_ms <= 0:
        raise ValueError(f"window must be positive, got {window_ms}")
    if window_ms % granularity_ms != 0:
        # The resolution ladder already refuses this at definition time. Repeating the
        # check here keeps the module usable on its own and makes the invariant local.
        raise ValueError(
            f"window {window_ms}ms is not a whole multiple of granularity {granularity_ms}ms"
        )
    head_start_ms = align_down(as_of_ms, granularity_ms)
    return WindowBounds(
        as_of_ms=as_of_ms,
        window_ms=window_ms,
        granularity_ms=granularity_ms,
        tile_start_index=tile_index(as_of_ms - window_ms, granularity_ms),
        tile_end_index=head_start_ms // granularity_ms,
        head_start_ms=head_start_ms,
    )


def retention_start_index(as_of_ms: int, retention_ms: int, granularity_ms: int) -> int:
    """The oldest tile index still readable at ``as_of_ms``.

    Anything below this can never contribute to any window on this grid again, which is
    what the online store expires on and what the tile compaction job deletes.
    """
    return tile_index(as_of_ms - retention_ms, granularity_ms)
