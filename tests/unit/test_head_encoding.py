"""The head ZSET member encoding: plain JSON, no Redis needed.

This is the contract the Kafka-to-Redis consumer must arrive at too, so these tests pin
down the exact shape rather than just "it round trips" -- including the ``event_id``
discriminator that keeps two events with identical column values from colliding in the
ZSET (see ``TestCollisionSafety``).
"""

from __future__ import annotations

import json

from asofline.online.head import decode_head_event, encode_head_event


class TestRoundTrip:
    def test_round_trips_multiple_columns(self) -> None:
        member = encode_head_event("e1", {"watch_seconds": 12.5, "liked": 0.0, "shared": 1.0})
        assert decode_head_event(member) == {
            "watch_seconds": 12.5,
            "liked": 0.0,
            "shared": 1.0,
        }

    def test_round_trips_a_null_column_value(self) -> None:
        """A ``watch_seconds`` column on a non-watch event, exactly like the raw event."""
        member = encode_head_event("e1", {"watch_seconds": None, "liked": 0.0})
        assert decode_head_event(member) == {"watch_seconds": None, "liked": 0.0}

    def test_empty_columns_round_trips(self) -> None:
        assert decode_head_event(encode_head_event("e1", {})) == {}

    def test_single_column_round_trips(self) -> None:
        assert decode_head_event(encode_head_event("e1", {"watch_seconds": 3.0})) == {
            "watch_seconds": 3.0
        }


class TestExactShape:
    def test_encoding_is_plain_json_with_sorted_keys(self) -> None:
        member = encode_head_event("e1", {"shared": 1.0, "liked": 0.0})
        assert member == json.dumps(
            {"_event_id": "e1", "liked": 0.0, "shared": 1.0}, sort_keys=True
        )

    def test_null_encodes_as_json_null_not_a_sentinel_string(self) -> None:
        member = encode_head_event("e1", {"watch_seconds": None})
        assert json.loads(member)["watch_seconds"] is None


class TestCollisionSafety:
    """The reason ``event_id`` exists at all: two identical-looking events must not merge.

    Every impression in this project's demo data has identical columns (``watch_seconds``
    null, ``liked`` 0, ``shared`` 0), and the head is a Redis ZSET, whose members are its
    identity: ``ZADD`` on an existing member updates its score rather than adding a second
    entry. Without a discriminator, two impressions in one head window would collapse to
    one entry and every aggregation reading the head would undercount. This was found by
    P5's skew detector against a live consumer before it was ever pinned down as a unit
    test here.
    """

    def test_identical_columns_with_different_event_ids_encode_differently(self) -> None:
        columns = {"watch_seconds": None, "liked": 0.0, "shared": 0.0}
        first = encode_head_event("event-a", columns)
        second = encode_head_event("event-b", columns)
        assert first != second

    def test_the_event_id_never_appears_in_the_decoded_columns(self) -> None:
        member = encode_head_event("event-a", {"watch_seconds": 1.0})
        decoded = decode_head_event(member)
        assert "_event_id" not in decoded
        assert decoded == {"watch_seconds": 1.0}


class TestDecodeTolerance:
    def test_a_json_integer_decodes_as_a_float(self) -> None:
        """A second, independent encoder might reasonably emit ``1`` rather than ``1.0``
        for an integer-valued column like ``liked``. Every caller hands the decoded value
        straight to ``Monoid.lift``, which expects a float, so this must not surface as an
        ``int`` here even though the JSON module would happily hand one back."""
        member = json.dumps({"liked": 1})
        assert decode_head_event(member) == {"liked": 1.0}
        assert isinstance(decode_head_event(member)["liked"], float)

    def test_decoding_does_not_reorder_or_drop_keys(self) -> None:
        member = json.dumps({"b": 2.0, "a": 1.0})
        assert decode_head_event(member) == {"a": 1.0, "b": 2.0}
