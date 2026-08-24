"""Partial aggregate states and window rollup.

Together with ``asofline.definitions`` this is the pure core: no Spark, no Redis, no
Kafka, no network. Both the batch path and the streaming path call into it, which is what
makes the shared-definition claim real. ``tests/unit/test_layering.py`` enforces it.
"""

from asofline.agg.monoid import (
    AVG,
    COUNT,
    MAX,
    MIN,
    MONOIDS,
    SUM,
    Monoid,
    State,
    merge_all,
    monoid_for,
)
from asofline.agg.rollup import brute_force, rollup, rollup_at, tiles_from_events
from asofline.agg.window import (
    WindowBounds,
    align_down,
    bounds_for,
    retention_start_index,
    tile_index,
)

__all__ = [
    "AVG",
    "COUNT",
    "MAX",
    "MIN",
    "MONOIDS",
    "SUM",
    "Monoid",
    "State",
    "WindowBounds",
    "align_down",
    "bounds_for",
    "brute_force",
    "merge_all",
    "monoid_for",
    "retention_start_index",
    "rollup",
    "rollup_at",
    "tile_index",
    "tiles_from_events",
]
