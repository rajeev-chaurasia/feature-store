"""Rolling tiles plus a head into one served value.

This is the function the whole "one definition, two paths" claim rests on. The batch
compiler calls it, the online serving layer calls it, and the skew detector compares their
outputs. If any caller were to reimplement it, the shared-definition argument would be
decorative.

It is deliberately free of I/O, Spark and Redis, so ``tests/unit`` exercises it directly
with no JVM and no containers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from asofline.agg.monoid import Monoid, State
from asofline.agg.window import WindowBounds, bounds_for


def rollup(
    monoid: Monoid,
    tiles: Mapping[int, State],
    head_events: Iterable[tuple[int, float]],
    bounds: WindowBounds,
) -> float | None:
    """Merge the covered tiles and the head, then finalize.

    ``tiles`` maps tile index to partial state and may contain indices outside the window;
    they are filtered rather than assumed absent, because the online store holds a whole
    grid's retention and only the window decides what counts.

    ``head_events`` is ``(event_ts_ms, value)`` pairs. For ``COUNT`` the value is ignored.

    Merge order is fixed: tiles ascending by index, then head events ascending by
    timestamp. Float addition is not associative, so an unspecified order would make the
    two paths disagree in the last bits and hand the skew detector a permanent,
    meaningless finding.
    """
    accumulator = monoid.identity
    for index in sorted(index for index in tiles if bounds.covers_tile(index)):
        accumulator = monoid.merge(accumulator, tiles[index])
    for _, value in sorted(
        (ts, value) for ts, value in head_events if bounds.covers_head_event(ts)
    ):
        accumulator = monoid.merge(accumulator, monoid.lift(value))
    return monoid.finalize(accumulator)


def rollup_at(
    monoid: Monoid,
    tiles: Mapping[int, State],
    head_events: Iterable[tuple[int, float]],
    *,
    as_of_ms: int,
    window_ms: int,
    granularity_ms: int,
) -> float | None:
    """``rollup`` with the bounds computed for you. The convenience form for callers."""
    return rollup(monoid, tiles, head_events, bounds_for(as_of_ms, window_ms, granularity_ms))


def tiles_from_events(
    monoid: Monoid,
    events: Iterable[tuple[int, float]],
    granularity_ms: int,
) -> dict[int, State]:
    """Bin ``(event_ts_ms, value)`` pairs into partial states, one per tile.

    The reference implementation of tiling. The Spark job produces the same mapping at
    scale, and ``tests/unit`` uses this to check that it does.
    """
    from asofline.agg.window import tile_index

    binned: dict[int, State] = {}
    for event_ts_ms, value in sorted(events):
        index = tile_index(event_ts_ms, granularity_ms)
        current = binned.get(index, monoid.identity)
        binned[index] = monoid.merge(current, monoid.lift(value))
    return binned


def brute_force(
    monoid: Monoid,
    events: Iterable[tuple[int, float]],
    *,
    as_of_ms: int,
    window_ms: int,
    granularity_ms: int,
) -> float | None:
    """The answer computed straight from raw events, with no tiles involved.

    This is the oracle. It applies the same snapped-trailing-edge rule as ``bounds_for``,
    so it checks that tiling is a faithful refactoring of the definition rather than
    checking a different definition. A version that used the exact ``[T - W, T)`` interval
    would disagree with any correct tiled implementation, and the disagreement would be
    the snapping, not a bug.
    """
    bounds = bounds_for(as_of_ms, window_ms, granularity_ms)
    accumulator = monoid.identity
    for event_ts_ms, value in sorted(events):
        if bounds.effective_start_ms <= event_ts_ms < bounds.as_of_ms:
            accumulator = monoid.merge(accumulator, monoid.lift(value))
    return monoid.finalize(accumulator)
