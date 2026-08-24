"""A feature view: one source, one set of entities, a bundle of windowed aggregations.

This is the single declaration that the batch compiler and the streaming compiler both
read. Nothing below this layer is allowed to hold an opinion about window semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from asofline.definitions.aggregation import Aggregation
from asofline.definitions.entity import Entity
from asofline.definitions.errors import DefinitionError
from asofline.definitions.naming import validate_identifier
from asofline.definitions.resolution import FIVE_MINUTE_RESOLUTION, Resolution
from asofline.definitions.source import EventSource


@dataclass(frozen=True, slots=True)
class FeatureView:
    """A named group of features over one event source.

    ``ttl`` is an online staleness bound, not a window: a value whose newest contributing
    event is older than ``ttl`` is served as null rather than as a stale number, and the
    offline path applies the identical rule so the two agree. It must be at least the
    longest window, because a shorter ttl would null out a value the window is still
    entitled to see.

    ``version`` is part of the online key. Bumping it gives a redefinition its own key
    space instead of mixing two incompatible encodings under one key.
    """

    name: str
    entities: tuple[Entity, ...]
    aggregations: tuple[Aggregation, ...]
    source: EventSource
    ttl: timedelta
    version: int = 1
    resolution: Resolution = field(default=FIVE_MINUTE_RESOLUTION)
    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.name, kind="feature view name")
        if self.version < 1:
            raise DefinitionError(f"{self.name}: version must be >= 1, got {self.version}")
        self._validate_entities()
        self._validate_aggregations()
        self._validate_ttl()

    def _validate_entities(self) -> None:
        if not self.entities:
            raise DefinitionError(f"{self.name}: a feature view needs at least one entity")
        join_keys = [entity.join_key for entity in self.entities]
        if len(set(join_keys)) != len(join_keys):
            raise DefinitionError(f"{self.name}: repeated join key in {join_keys}")

    def _validate_aggregations(self) -> None:
        if not self.aggregations:
            raise DefinitionError(f"{self.name}: a feature view needs at least one aggregation")
        seen: set[str] = set()
        for aggregation in self.aggregations:
            for window in aggregation.windows:
                # Raises if the resolution ladder cannot express this window.
                self.resolution.granularity_for(window)
            for feature_name in aggregation.feature_names:
                if feature_name in seen:
                    raise DefinitionError(f"{self.name}: duplicate feature {feature_name!r}")
                seen.add(feature_name)

    def _validate_ttl(self) -> None:
        if self.ttl <= timedelta(0):
            raise DefinitionError(f"{self.name}: ttl must be positive, got {self.ttl!r}")
        if self.ttl < self.longest_window:
            raise DefinitionError(
                f"{self.name}: ttl {self.ttl!r} is shorter than the longest window "
                f"{self.longest_window!r}, which would null out values the window still needs"
            )

    @property
    def join_keys(self) -> tuple[str, ...]:
        return tuple(entity.join_key for entity in self.entities)

    @property
    def windows(self) -> tuple[timedelta, ...]:
        seen: set[timedelta] = set()
        for aggregation in self.aggregations:
            seen.update(aggregation.windows)
        return tuple(sorted(seen))

    @property
    def longest_window(self) -> timedelta:
        return max(self.windows)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for aggregation in self.aggregations:
            names.extend(aggregation.feature_names)
        return tuple(names)

    @property
    def qualified_feature_names(self) -> tuple[str, ...]:
        """Feature names as the serving API exposes them, ``view:feature``.

        Two views may legitimately aggregate the same source column the same way, so the
        short name is unique only within a view. Inside the view's Iceberg table and its
        Redis hash the view already namespaces, and the short name is what is stored.
        Only the flat serving response needs the prefix, so only it carries the cost.
        """
        return tuple(f"{self.name}:{name}" for name in self.feature_names)

    @property
    def grids(self) -> tuple[timedelta, ...]:
        """The distinct tile granularities this view actually writes to, finest first.

        The streaming consumer updates one field per grid per event, so this is directly
        the per-event write fan-out.
        """
        return tuple(sorted({self.resolution.granularity_for(w) for w in self.windows}))

    def retention_for(self, granularity: timedelta) -> timedelta:
        """How far back tiles on one grid must be kept.

        Anything older than the longest window answered by that grid can never be read
        again, so this is what the online store expires on.
        """
        windows = [w for w in self.windows if self.resolution.granularity_for(w) == granularity]
        if not windows:
            raise DefinitionError(f"{self.name}: no window uses granularity {granularity!r}")
        return max(windows)
