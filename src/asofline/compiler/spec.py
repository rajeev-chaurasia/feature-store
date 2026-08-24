"""What a definition compiles down to, in a form both runtimes can read.

A ``TileSpec`` is one ``(aggregation, grid)`` pair: the unit of tile state. Windows do not
appear in it, because every window of an aggregation that resolves to the same grid reads
the same tiles. That collapsing is the reason a view with three windows does not cost
three times the storage.

A ``FeatureSpec`` is one served feature: an aggregation, a window, and the grid that
window resolves to.

Both are plain data. The Spark compiler turns specs into SQL; the streaming consumer turns
the same specs into Redis writes. Neither one re-derives which grid a window belongs to,
so they cannot disagree about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from asofline.definitions.aggregation import AggFunction, Aggregation
from asofline.definitions.view import FeatureView


def _millis(value: timedelta) -> int:
    return int(value.total_seconds() * 1000)


@dataclass(frozen=True, slots=True)
class TileSpec:
    """One unit of tile state: an aggregation on one grid."""

    view_name: str
    view_version: int
    agg_name: str
    function: AggFunction
    column: str | None
    granularity_ms: int
    retention_ms: int

    @property
    def arity(self) -> int:
        return 2 if self.function is AggFunction.AVG else 1


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One served feature."""

    view_name: str
    view_version: int
    feature_name: str
    agg_name: str
    function: AggFunction
    column: str | None
    window_ms: int
    granularity_ms: int

    @property
    def qualified_name(self) -> str:
        return f"{self.view_name}:{self.feature_name}"

    @property
    def tile_key(self) -> tuple[str, int]:
        """Which ``TileSpec`` this feature reads, as ``(agg_name, granularity_ms)``."""
        return (self.agg_name, self.granularity_ms)


def feature_specs(view: FeatureView) -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for aggregation in view.aggregations:
        for window in aggregation.windows:
            specs.append(
                FeatureSpec(
                    view_name=view.name,
                    view_version=view.version,
                    feature_name=aggregation.feature_name(window),
                    agg_name=aggregation.basename,
                    function=aggregation.function,
                    column=aggregation.column,
                    window_ms=_millis(window),
                    granularity_ms=_millis(view.resolution.granularity_for(window)),
                )
            )
    return tuple(specs)


def tile_specs(view: FeatureView) -> tuple[TileSpec, ...]:
    """Deduplicated ``(aggregation, grid)`` pairs, with each one's retention.

    Retention is the longest window that reads this grid. Anything older can never
    contribute again, which is what the online store expires on and what tile compaction
    deletes.
    """
    longest: dict[tuple[str, int], int] = {}
    source: dict[tuple[str, int], tuple[Aggregation, int]] = {}
    for aggregation in view.aggregations:
        for window in aggregation.windows:
            granularity_ms = _millis(view.resolution.granularity_for(window))
            key = (aggregation.basename, granularity_ms)
            window_ms = _millis(window)
            longest[key] = max(longest.get(key, 0), window_ms)
            source[key] = (aggregation, granularity_ms)

    specs: list[TileSpec] = []
    for key in sorted(longest):
        aggregation, granularity_ms = source[key]
        specs.append(
            TileSpec(
                view_name=view.name,
                view_version=view.version,
                agg_name=aggregation.basename,
                function=aggregation.function,
                column=aggregation.column,
                granularity_ms=granularity_ms,
                retention_ms=longest[key],
            )
        )
    return tuple(specs)
