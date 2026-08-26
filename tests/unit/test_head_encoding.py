"""The head ZSET member encoding: plain JSON, no Redis needed.

This is the contract the (parallel, independently built) Kafka-to-Redis consumer must
arrive at too, so these tests pin down the exact shape rather than just "it round trips".
"""

from __future__ import annotations

import json

from asofline.online.head import decode_head_event, encode_head_event


class TestRoundTrip:
    def test_round_trips_multiple_columns(self) -> None:
        member = encode_head_event({"watch_seconds": 12.5, "liked": 0.0, "shared": 1.0})
        assert decode_head_event(member) == {
            "watch_seconds": 12.5,
            "liked": 0.0,
            "shared": 1.0,
        }

    def test_round_trips_a_null_column_value(self) -> None:
        """A ``watch_seconds`` column on a non-watch event, exactly like the raw event."""
        member = encode_head_event({"watch_seconds": None, "liked": 0.0})
        assert decode_head_event(member) == {"watch_seconds": None, "liked": 0.0}

    def test_empty_columns_round_trips(self) -> None:
        assert decode_head_event(encode_head_event({})) == {}

    def test_single_column_round_trips(self) -> None:
        assert decode_head_event(encode_head_event({"watch_seconds": 3.0})) == {
            "watch_seconds": 3.0
        }


class TestExactShape:
    def test_encoding_is_plain_json_with_sorted_keys(self) -> None:
        member = encode_head_event({"shared": 1.0, "liked": 0.0})
        assert member == json.dumps({"liked": 0.0, "shared": 1.0}, sort_keys=True)

    def test_null_encodes_as_json_null_not_a_sentinel_string(self) -> None:
        member = encode_head_event({"watch_seconds": None})
        assert json.loads(member) == {"watch_seconds": None}


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
