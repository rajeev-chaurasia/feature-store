"""Aggregation functions and the algebra that decides how each one is computed.

Only aggregations that form a commutative monoid are supported, because every value
this store serves is assembled by merging precomputed tiles. That constraint is the
price of having one definition drive both the batch and the streaming path, and it is
why exact median and exact distinct count are absent rather than merely unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from asofline.definitions.errors import DefinitionError


class AggFunction(StrEnum):
    """A mergeable aggregation.

    ``SUM``, ``COUNT`` and ``AVG`` additionally form a group: every partial state has an
    inverse, so a window can be answered by subtracting two prefix sums. ``MIN`` and
    ``MAX`` have no inverse and need the range path instead. ``has_inverse`` is what the
    batch compiler branches on.
    """

    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    AVG = "avg"

    @property
    def has_inverse(self) -> bool:
        return self in (AggFunction.SUM, AggFunction.COUNT, AggFunction.AVG)

    @property
    def reads_source_column(self) -> bool:
        """``COUNT`` counts rows, so it is the one function with no source column."""
        return self is not AggFunction.COUNT


def format_window(window: timedelta) -> str:
    """Render a window as the compact label used in feature names.

    Feature names end up in Redis hash fields, Iceberg column names and HTTP responses,
    so the label has to be stable and free of characters those three disagree about.
    """
    total = int(window.total_seconds())
    if total <= 0:
        raise DefinitionError(f"window must be positive, got {window!r}")
    for unit_seconds, suffix in ((86_400, "d"), (3_600, "h"), (60, "m"), (1, "s")):
        if total % unit_seconds == 0:
            return f"{total // unit_seconds}{suffix}"
    raise DefinitionError(f"window {window!r} is not a whole number of seconds")


@dataclass(frozen=True, slots=True)
class Aggregation:
    """One aggregation of one source column over one or more trailing windows.

    Each ``(function, column, window)`` triple becomes a separate served feature. The
    windows are grouped here rather than declared separately because they share a source
    column and therefore share tile state on the fine or coarse grid.
    """

    function: AggFunction
    windows: tuple[timedelta, ...]
    column: str | None = None

    def __post_init__(self) -> None:
        if not self.windows:
            raise DefinitionError(f"{self.function} declares no windows")
        if len(set(self.windows)) != len(self.windows):
            raise DefinitionError(f"{self.function} repeats a window: {self.windows}")
        if self.function.reads_source_column and not self.column:
            raise DefinitionError(f"{self.function} needs a source column")
        if not self.function.reads_source_column and self.column:
            raise DefinitionError(
                f"{self.function} counts rows and must not name a source column, "
                f"got {self.column!r}"
            )
        for window in self.windows:
            format_window(window)

    @property
    def basename(self) -> str:
        return f"{self.column}_{self.function}" if self.column else str(self.function)

    def feature_name(self, window: timedelta) -> str:
        """The served name for one window, for example ``watch_seconds_sum_24h``."""
        if window not in self.windows:
            raise DefinitionError(f"{window!r} is not a window of {self.basename}")
        return f"{self.basename}_{format_window(window)}"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_name(window) for window in self.windows)
