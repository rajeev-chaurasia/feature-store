"""Redis key and field naming, and the entity-key encoding both sides must agree on.

One hash per ``(view, version, entity value, grid)``, not one per ``(view, entity, agg,
grid)``. Every aggregation on the same grid for the same entity is a field in the same
hash, which is what makes one read of a grid a single ``HGETALL`` rather than one round
trip per aggregation. The version is embedded in the key, per ``FeatureView.version``, so
redefining a view's aggregations does not require migrating or invalidating old data: the
old version's keys simply age out under their own TTL while the new version starts clean.
"""

from __future__ import annotations

from asofline.definitions.view import FeatureView

_UNIT_SEPARATOR = "\x1f"
"""ASCII unit separator. Chosen over a printable character like ``|`` because a join key
value (a user id, a video id) is far more likely to contain a pipe than this control
character, and an ambiguous join risks merging two different entities' state."""

TILE_HASH_PREFIX = "fs:tiles"
HEAD_ZSET_PREFIX = "fs:head"


def entity_value_key(view: FeatureView, values: dict[str, str]) -> str:
    """Encode a join-key value tuple in the view's declared key order.

    Ordering by ``view.join_keys`` rather than by the order ``values`` happens to iterate
    in is what makes this deterministic: a dict built from a Kafka message and a dict
    built from an HTTP request body have no guaranteed order of their own.
    """
    missing = [key for key in view.join_keys if key not in values]
    if missing:
        raise KeyError(f"{view.name}: missing join key value(s) {missing}")
    return _UNIT_SEPARATOR.join(values[key] for key in view.join_keys)


def tile_hash_key(view: FeatureView, entity_key: str, granularity_ms: int) -> str:
    return f"{TILE_HASH_PREFIX}:{view.name}:v{view.version}:{granularity_ms}:{entity_key}"


def tile_field(agg_name: str, tile_index: int) -> str:
    return f"{agg_name}:{tile_index}"


def parse_tile_field(field: str) -> tuple[str, int]:
    agg_name, _, index = field.rpartition(":")
    if not agg_name:
        raise ValueError(f"not a tile field: {field!r}")
    return agg_name, int(index)


def head_zset_key(view: FeatureView, entity_key: str) -> str:
    """One key per entity for the whole view, covering every grid.

    A single sorted set rather than one per grid: the head window is always a suffix of
    recent raw events regardless of which grid is asking, so the same stored events answer
    every grid's head query, filtered at read time by ``asofline.agg.window`` bounds.
    """
    return f"{HEAD_ZSET_PREFIX}:{view.name}:v{view.version}:{entity_key}"
