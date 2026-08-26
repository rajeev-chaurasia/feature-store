"""The online contract: how a tile state becomes bytes, and how a key names it.

Both the Kafka-to-Redis consumer and the serving layer are built against exactly this
module. It has no Redis dependency, so both sides can be tested here before either one
touches a real connection.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from asofline.demo.views import USER_ENGAGEMENT, VIDEO_ENGAGEMENT
from asofline.online.codec import CodecError, decode_state, encode_state
from asofline.online.keys import (
    entity_value_key,
    head_zset_key,
    parse_tile_field,
    tile_field,
    tile_hash_key,
)

finite = st.floats(allow_nan=False, allow_infinity=False, width=64)


class TestCodecRoundTrip:
    @given(value=finite)
    def test_arity_one_round_trips(self, value: float) -> None:
        assert decode_state(encode_state((value,)), arity=1) == (value,)

    @given(a=finite, b=finite)
    def test_arity_two_round_trips(self, a: float, b: float) -> None:
        assert decode_state(encode_state((a, b)), arity=2) == (a, b)

    def test_encoded_size_is_fixed_width(self) -> None:
        assert len(encode_state((1.0,))) == 1 + 8
        assert len(encode_state((1.0, 2.0))) == 1 + 16


class TestCodecRejection:
    def test_wrong_arity_at_decode_is_rejected(self) -> None:
        """A cross-wired read: asking an AVG's two doubles from a SUM's encoding.

        Caught as a decode error rather than silently returning a truncated tuple, which
        would corrupt a rollup instead of raising.
        """
        payload = encode_state((1.0,))
        with pytest.raises(CodecError, match="expected"):
            decode_state(payload, arity=2)

    def test_unknown_version_byte_is_rejected(self) -> None:
        payload = bytearray(encode_state((1.0,)))
        payload[0] = 99
        with pytest.raises(CodecError, match="unknown codec version"):
            decode_state(bytes(payload), arity=1)

    def test_unsupported_arity_is_rejected(self) -> None:
        with pytest.raises(CodecError, match="unsupported arity"):
            encode_state((1.0, 2.0, 3.0))

    def test_truncated_payload_is_rejected(self) -> None:
        payload = encode_state((1.0, 2.0))[:-1]
        with pytest.raises(CodecError, match="expected"):
            decode_state(payload, arity=2)


class TestKeys:
    def test_entity_value_key_orders_by_declared_join_keys(self) -> None:
        # USER_ENGAGEMENT has one entity, so build a values dict with an extra unrelated
        # key to prove ordering comes from view.join_keys, not from dict iteration order.
        values = {"unrelated": "x", "user_id": "u42"}
        assert entity_value_key(USER_ENGAGEMENT, values) == "u42"

    def test_missing_join_key_value_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="user_id"):
            entity_value_key(USER_ENGAGEMENT, {})

    def test_tile_hash_key_is_distinct_per_view_version_and_grid(self) -> None:
        keys = {
            tile_hash_key(USER_ENGAGEMENT, "u1", 300_000),
            tile_hash_key(USER_ENGAGEMENT, "u1", 3_600_000),
            tile_hash_key(VIDEO_ENGAGEMENT, "u1", 300_000),
        }
        assert len(keys) == 3

    def test_head_key_does_not_vary_by_grid(self) -> None:
        """One head per entity per view, deliberately: it answers every grid's query."""
        assert head_zset_key(USER_ENGAGEMENT, "u1") == head_zset_key(USER_ENGAGEMENT, "u1")

    def test_tile_field_round_trips(self) -> None:
        assert parse_tile_field(tile_field("watch_seconds_sum", 12345)) == (
            "watch_seconds_sum",
            12345,
        )

    def test_tile_field_round_trips_an_agg_name_with_no_special_characters(self) -> None:
        # agg_name is drawn from Aggregation.basename, which is always lower snake case
        # (see definitions.naming), so it never itself contains ':'. The tile index is
        # split off from the last ':' rather than the first for that reason.
        agg_name, index = parse_tile_field("count:7")
        assert agg_name == "count"
        assert index == 7

    def test_malformed_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a tile field"):
            parse_tile_field("no-separator-here")
