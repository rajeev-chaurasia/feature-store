"""The online store's read side: tiles plus head, pipelined, rolled up client-side.

One Redis round trip per request, no matter how many entities it asks for. Every entity's
``HGETALL`` (one per grid the view uses) and its head ``ZRANGEBYSCORE`` are queued onto one
``redis.asyncio.Redis().pipeline()`` before anything is sent, then ``execute()`` is called
exactly once. Decoding and the call into ``asofline.agg.rollup.rollup_at`` happen after
that, entirely client-side, with no further Redis traffic. Issuing one round trip per
entity would defeat the reason pipelining exists, and the plan calls this out explicitly
as the performance requirement this module has to meet.

**Freshness.** An entity's most recent visible event can live in the head ZSET (if it is
still inside the current, unfinished tile on some grid) or only in a completed tile (if a
writer has since trimmed the head down to the current tile's start, which is the whole
point of the head existing only to make the leading edge exact rather than as a general
event log). So "last seen" is taken as the later of: the newest head event actually read,
and the start of the most recent tile present on any grid. The tile-start figure
underestimates the true last-seen time by at most one grid's granularity, which biases
toward calling an entity stale a little early rather than a little late, matching this
project's general preference for nulling out over quietly serving something wrong. An
entity with neither head events nor any tile data has never been seen at all and is served
nulls unconditionally, matching ``asofline.offline.pit``'s treatment of an unseen entity.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from redis.asyncio import Redis

from asofline.agg.monoid import State, monoid_for
from asofline.agg.rollup import rollup_at
from asofline.agg.window import align_down
from asofline.compiler.spec import FeatureSpec, TileSpec, feature_specs, tile_specs
from asofline.definitions.view import FeatureView
from asofline.online.codec import decode_state
from asofline.online.head import decode_head_event
from asofline.online.keys import entity_value_key, head_zset_key, parse_tile_field, tile_hash_key

FeatureVector = dict[str, float | None]

_HashResult = Mapping[bytes, bytes]
_HeadResult = Sequence[tuple[bytes, float]]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _decode_text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _head_events_for_spec(
    head_events: Sequence[tuple[int, dict[str, float | None]]], spec: FeatureSpec
) -> list[tuple[int, float]]:
    """Which of an entity's head events count toward one served feature.

    ``COUNT`` has no column and counts every event on the head regardless of its columns.
    Every other function only counts events that actually carry its column: an event whose
    JSON has that column as ``null``, or omits it entirely, is one that never touched this
    aggregation, and including it would fold a stray ``0.0`` (or worse, a KeyError) into
    the wrong sum.
    """
    if spec.column is None:
        return [(ts, 0.0) for ts, _ in head_events]
    result: list[tuple[int, float]] = []
    for ts, columns in head_events:
        value = columns.get(spec.column)
        if value is not None:
            result.append((ts, value))
    return result


class OnlineStore:
    """An async Redis client wrapper answering online feature-vector reads."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @classmethod
    def from_url(cls, url: str) -> OnlineStore:
        return cls(Redis.from_url(url))

    async def close(self) -> None:
        await self._redis.aclose()

    async def get_online_features(
        self,
        view: FeatureView,
        entities: Sequence[Mapping[str, str]],
        *,
        as_of_ms: int | None = None,
    ) -> list[FeatureVector]:
        """One feature vector per entity, in the same order as ``entities``.

        ``as_of_ms`` defaults to "now" but is injectable so tests are deterministic.
        """
        resolved_as_of_ms = _now_ms() if as_of_ms is None else as_of_ms
        if not entities:
            return []

        specs = feature_specs(view)
        all_tile_specs = tile_specs(view)
        # Keyed by agg_name alone for arity lookup only: arity depends on the aggregation
        # function, not the grid, so collapsing here is safe. The set of grids the view
        # actually writes to is not safe to collapse this way, because one aggregation
        # (for example ``watch_seconds_sum``, spanning a 1h and a 7d window) can span more
        # than one grid, so it is computed from every spec rather than from this dict.
        tile_lookup: dict[str, TileSpec] = {spec.agg_name: spec for spec in all_tile_specs}
        grids = sorted({spec.granularity_ms for spec in all_tile_specs})
        ttl_ms = int(view.ttl.total_seconds() * 1000)
        # The coarsest grid's head boundary is the earliest timestamp any grid's head
        # query could need (asofline.agg.window.bounds_for): a finer grid's own head
        # starts later, or at the same instant, so this one bound covers every grid at
        # once and there is no need to fetch every event the ZSET has ever held.
        head_start_ms = align_down(resolved_as_of_ms, grids[-1])

        entity_keys = [entity_value_key(view, dict(values)) for values in entities]

        pipeline = self._redis.pipeline(transaction=False)
        for entity_key in entity_keys:
            for granularity_ms in grids:
                pipeline.hgetall(tile_hash_key(view, entity_key, granularity_ms))
            pipeline.zrangebyscore(
                head_zset_key(view, entity_key),
                head_start_ms,
                resolved_as_of_ms,
                withscores=True,
            )
        raw_results = await pipeline.execute()

        stride = len(grids) + 1
        vectors: list[FeatureVector] = []
        for row_index in range(len(entities)):
            chunk = raw_results[row_index * stride : (row_index + 1) * stride]
            hash_results: list[_HashResult] = chunk[: len(grids)]
            head_raw: _HeadResult = chunk[len(grids)]
            vectors.append(
                self._roll_up_one(
                    specs,
                    tile_lookup,
                    grids,
                    hash_results,
                    head_raw,
                    as_of_ms=resolved_as_of_ms,
                    ttl_ms=ttl_ms,
                )
            )
        return vectors

    def _roll_up_one(
        self,
        specs: tuple[FeatureSpec, ...],
        tile_lookup: Mapping[str, TileSpec],
        grids: Sequence[int],
        hash_results: Sequence[_HashResult],
        head_raw: _HeadResult,
        *,
        as_of_ms: int,
        ttl_ms: int,
    ) -> FeatureVector:
        tiles_by_key: dict[tuple[str, int], dict[int, State]] = {}
        last_seen_candidates: list[int] = []

        for granularity_ms, hash_result in zip(grids, hash_results, strict=True):
            indices_seen: list[int] = []
            for field_raw, payload in hash_result.items():
                agg_name, tile_idx = parse_tile_field(_decode_text(field_raw))
                tile_spec = tile_lookup[agg_name]
                state = decode_state(payload, arity=tile_spec.arity)
                tiles_by_key.setdefault((agg_name, granularity_ms), {})[tile_idx] = state
                indices_seen.append(tile_idx)
            if indices_seen:
                # A tile's start is a lower bound on this entity's last activity on this
                # grid. See the module docstring for why the head ZSET alone cannot be
                # trusted to carry the full freshness signal.
                last_seen_candidates.append(max(indices_seen) * granularity_ms)

        head_events: list[tuple[int, dict[str, float | None]]] = []
        for member_raw, score in head_raw:
            head_events.append((int(score), decode_head_event(_decode_text(member_raw))))
        if head_events:
            last_seen_candidates.append(max(ts for ts, _ in head_events))

        last_seen_ms = max(last_seen_candidates) if last_seen_candidates else None
        is_fresh = last_seen_ms is not None and (as_of_ms - last_seen_ms) <= ttl_ms

        vector: FeatureVector = {}
        for spec in specs:
            if not is_fresh:
                vector[spec.feature_name] = None
                continue
            monoid = monoid_for(spec.function)
            tiles = tiles_by_key.get(spec.tile_key, {})
            relevant_head = _head_events_for_spec(head_events, spec)
            vector[spec.feature_name] = rollup_at(
                monoid,
                tiles,
                relevant_head,
                as_of_ms=as_of_ms,
                window_ms=spec.window_ms,
                granularity_ms=spec.granularity_ms,
            )
        return vector
