"""How a raw event becomes a head ZSET member.

The head ZSET (``online.keys.head_zset_key``) holds recent raw events so an in-progress
tile's leading edge can be answered exactly (the leading-edge-exact rule lives in
``agg.window``). The ZSET's score is the event's ``event_ts_ms``, so this module only has
to define the *member*: how one event's column values are packed into a string.

**Format: a plain JSON object, standard library ``json``, keys sorted.** Not the tile
codec's fixed-width binary: a head event has to be read back by whichever feature later
asks for one of its columns, keyed by column name, and the obvious way to store "the
columns one raw event carries" is the same shape ``asofline.demo.events.EngagementEvent``
already uses for them: a mapping from column name to value, ``None`` where the event does
not carry that column at all. A ``watch`` event has a real ``watch_seconds``; an
``impression`` event has ``None`` there, exactly like the source dataclass. This is
deliberately readable straight off ``FeatureSpec.column`` / ``Aggregation.column`` with no
private knowledge of this module: a column name that is ``None`` (``COUNT``) never appears
as a key, and every other column name a view declares maps to a float or ``null``.

``event_ts_ms`` is deliberately not a key in this object: it is already the ZSET score, and
storing it twice would let the two disagree.

**The member also carries the source event's ``event_id``, under the reserved key
``_event_id``.** This is not optional. A ZSET's members are its identity: ``ZADD`` on a
member that already exists updates its score rather than adding a second entry, and every
``impression`` event in this project's demo data encodes to the *identical* JSON object
(``watch_seconds`` null, ``liked`` 0, ``shared`` 0, no other columns) because impressions
never populate any of them. Without a discriminator, a user with two impressions in the
same head window silently collapses to one ZSET entry, and every aggregation that reads
the head, ``COUNT`` most visibly, undercounts by the number of collisions. This was found,
not anticipated: P5's skew detector's first real run against a live consumer flagged a
reproducible undercount concentrated on the most active synthetic users, which is exactly
who accumulates the most same-window duplicate impressions. ``_event_id`` is excluded from
what ``decode_head_event`` returns, so every existing caller, which wants only the
feature columns, is unaffected by its presence.

This is written down here, rather than only in the reader's code, because the
Kafka-to-Redis consumer (built independently, against this same module) is the writer, and
the two sides have to arrive at the same shape without coordinating directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

_EVENT_ID_KEY = "_event_id"


def encode_head_event(event_id: str, columns: Mapping[str, float | None]) -> str:
    """Pack one raw event's column values, keyed by column name, plus its ``event_id``.

    ``sort_keys=True`` so two encoders handed the same columns in a different dict order
    produce byte-identical members for the same event. That is not required for
    correctness (the ZSET is parsed back on read, never compared as bytes), but it removes
    one axis on which this implementation and the consumer's independent one could
    needlessly diverge.
    """
    body: dict[str, float | str | None] = {
        name: (None if value is None else float(value)) for name, value in columns.items()
    }
    body[_EVENT_ID_KEY] = event_id
    return json.dumps(body, sort_keys=True)


def decode_head_event(member: str) -> dict[str, float | None]:
    """Unpack a ZSET member back into its column map.

    ``_event_id`` is dropped here rather than left for callers to filter: every existing
    caller wants only the declared feature columns, keyed by ``FeatureSpec.column`` names,
    and none of those is ever ``_event_id``.

    Values come back as ``float`` (or ``None``), never ``int``, even if the encoder (or an
    independently written producer) emitted a JSON integer, because every caller hands
    this straight to ``Monoid.lift``, which expects a float.
    """
    decoded: dict[str, float | int | str | None] = json.loads(member)
    decoded.pop(_EVENT_ID_KEY, None)
    return {name: (None if value is None else float(value)) for name, value in decoded.items()}
