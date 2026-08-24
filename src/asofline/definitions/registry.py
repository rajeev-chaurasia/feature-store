"""The registry: every view the deployment knows about, validated as a whole.

Per-view validation lives in ``FeatureView``. What only the registry can check is
cross-view consistency, and the one that matters is feature name collisions. The serving
response is a flat map of feature name to value, so two views exporting the same feature
name is an ambiguity the API cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass

from asofline.definitions.entity import Entity
from asofline.definitions.errors import DefinitionError
from asofline.definitions.view import FeatureView


@dataclass(frozen=True, slots=True)
class Registry:
    views: tuple[FeatureView, ...]

    def __post_init__(self) -> None:
        if not self.views:
            raise DefinitionError("a registry needs at least one feature view")
        self._check_unique_view_names()
        self._check_unique_feature_names()
        self._check_entity_definitions_agree()

    def _check_unique_view_names(self) -> None:
        keys = [(view.name, view.version) for view in self.views]
        if len(set(keys)) != len(keys):
            raise DefinitionError(f"repeated (name, version) among views: {sorted(keys)}")

    def _check_unique_feature_names(self) -> None:
        """Qualified names must be unique across the whole registry.

        Unique ``(name, version)`` view keys almost imply this, but not quite: two
        versions of one view export the same qualified names, and the serving response
        cannot carry both. Asserting it directly is cheaper than reasoning about it.
        """
        owner: dict[str, str] = {}
        for view in self.views:
            label = f"{view.name} v{view.version}"
            for qualified in view.qualified_feature_names:
                previous = owner.get(qualified)
                if previous is not None:
                    raise DefinitionError(
                        f"feature {qualified!r} is exported by both {previous!r} and "
                        f"{label!r}; the serving response is a flat map and cannot carry both"
                    )
                owner[qualified] = label

    def _check_entity_definitions_agree(self) -> None:
        """Two views naming the same entity must mean the same key and type.

        Otherwise a caller passing ``user_id`` gets one view keyed on a string and
        another on an int, and only one of them finds anything.
        """
        known: dict[str, Entity] = {}
        for view in self.views:
            for entity in view.entities:
                previous = known.setdefault(entity.name, entity)
                if (previous.join_key, previous.key_type) != (entity.join_key, entity.key_type):
                    raise DefinitionError(
                        f"entity {entity.name!r} is defined inconsistently: "
                        f"{previous.join_key}/{previous.key_type} vs "
                        f"{entity.join_key}/{entity.key_type}"
                    )

    @property
    def entities(self) -> tuple[Entity, ...]:
        known: dict[str, Entity] = {}
        for view in self.views:
            for entity in view.entities:
                known.setdefault(entity.name, entity)
        return tuple(known[name] for name in sorted(known))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Every feature the registry serves, qualified as ``view:feature``."""
        names: list[str] = []
        for view in self.views:
            names.extend(view.qualified_feature_names)
        return tuple(names)

    def view(self, name: str, version: int | None = None) -> FeatureView:
        matches = [v for v in self.views if v.name == name]
        if not matches:
            raise DefinitionError(f"no feature view named {name!r}")
        if version is None:
            return max(matches, key=lambda v: v.version)
        for candidate in matches:
            if candidate.version == version:
                return candidate
        raise DefinitionError(f"no feature view {name!r} at version {version}")

    def view_for_feature(self, qualified_name: str) -> FeatureView:
        """Resolve ``view:feature``.

        The prefix is not trusted on its own: the feature must actually exist in the view
        it names, so a typo in the suffix fails here rather than returning a null the
        caller reads as "this entity has no history".
        """
        view_name, separator, short_name = qualified_name.partition(":")
        if not separator:
            raise DefinitionError(
                f"feature {qualified_name!r} is not qualified; expected 'view:feature'"
            )
        view = self.view(view_name)
        if short_name not in view.feature_names:
            raise DefinitionError(f"view {view_name!r} exports no feature {short_name!r}")
        return view
