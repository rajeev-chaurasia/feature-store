"""An independent recomputation of every feature, in pure Python, from raw events.

This is the oracle the Spark implementation is checked against. It shares only
``asofline.agg``, which ``tests/unit`` verifies against the definition directly, so it is
not the same code path with the same bugs. In particular it never touches the tile table,
the compiler, or Spark.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from asofline.agg import brute_force, monoid_for
from asofline.compiler.spec import FeatureSpec, feature_specs
from asofline.definitions.view import FeatureView
from asofline.demo.events import EngagementEvent

# How a source column name is read off an event. COUNT has no column and is handed a
# constant, matching Monoid.lift for COUNT ignoring its argument.
COLUMN_READERS: dict[str | None, Callable[[EngagementEvent], float | None]] = {
    None: lambda _event: 1.0,
    "watch_seconds": lambda event: event.watch_seconds,
    "liked": lambda event: float(event.liked),
    "shared": lambda event: float(event.shared),
}


def _stream(spec: FeatureSpec, events: Iterable[EngagementEvent]) -> list[tuple[int, float]]:
    """``(event_ts, value)`` pairs, with nulls dropped.

    Dropping nulls here is what makes ``avg`` mean seconds per watch. It mirrors SQL
    aggregate semantics, which is what the Spark path gets for free.
    """
    read = COLUMN_READERS[spec.column]
    pairs: list[tuple[int, float]] = []
    for event in events:
        value = read(event)
        if value is not None:
            pairs.append((event.event_ts, value))
    return pairs


def expected_features(
    view: FeatureView,
    events_by_entity: dict[str, list[EngagementEvent]],
    entity_key: str,
    as_of_ms: int,
) -> dict[str, float | None]:
    events = events_by_entity.get(entity_key, [])
    result: dict[str, float | None] = {}
    for spec in feature_specs(view):
        result[spec.feature_name] = brute_force(
            monoid_for(spec.function),
            _stream(spec, events),
            as_of_ms=as_of_ms,
            window_ms=spec.window_ms,
            granularity_ms=spec.granularity_ms,
        )
    return result


def group_by_entity(
    events: Iterable[EngagementEvent], join_key: str
) -> dict[str, list[EngagementEvent]]:
    grouped: dict[str, list[EngagementEvent]] = {}
    for event in events:
        grouped.setdefault(getattr(event, join_key), []).append(event)
    return grouped


def values_agree(left: float | None, right: float | None, *, rel: float = 1e-9) -> bool:
    """Null matches only null. Numbers match to a relative tolerance.

    The tolerance is not slack for a possibly wrong implementation. The prefix-sum
    strategy computes a window as the difference of two running totals, which is a
    different sequence of float operations from summing the covered tiles, so the last
    bits legitimately differ. ``compiler.batch`` documents this.
    """
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=rel, abs_tol=1e-9)
