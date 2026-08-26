"""How a tile state is packed into a Redis hash field.

This is the contract the Kafka-to-Redis consumer (a writer) and the serving layer (a
reader) must agree on exactly, so it is specified once here and imported by both rather
than each independently deciding how to pack a float tuple.

**Format: version byte + fixed-width doubles, big-endian.** Not JSON, not msgpack. A tile
update from the consumer is a hot path (one write per event per grid it touches), and a
serving read fans out to every tile a window covers, so the format is chosen to make
encode and decode cheap rather than to be self-describing. Every state this project has is
at most two doubles (``asofline.agg.monoid.MAX_STATE_ARITY`` mirrors this as
``MAX_STATE_ARITY`` in ``offline.tables``), so the wire format is fixed-width: one byte for
the version, then ``arity * 8`` bytes of big-endian IEEE 754 doubles.

**The version byte is checked on every decode, not assumed.** A schema change that altered
arity or byte order without bumping the version would silently misread old data as
garbage floats rather than failing loudly, and a garbage float folded into a sum is far
harder to notice than a decode error.
"""

from __future__ import annotations

import struct

CODEC_VERSION = 1
_HEADER = struct.Struct(">B")


class CodecError(ValueError):
    """A byte string is not a valid encoded tile state under this codec."""


def _body_format(arity: int) -> struct.Struct:
    if arity not in (1, 2):
        raise CodecError(f"unsupported arity {arity}; this codec packs 1 or 2 doubles")
    return struct.Struct(f">{arity}d")


def encode_state(state: tuple[float, ...]) -> bytes:
    """Pack a monoid state into its wire form, version byte first."""
    body = _body_format(len(state))
    return _HEADER.pack(CODEC_VERSION) + body.pack(*state)


def decode_state(payload: bytes, *, arity: int) -> tuple[float, ...]:
    """Unpack a wire-form tile state, verifying the version and the length.

    ``arity`` is supplied by the caller rather than inferred from the payload length,
    because the aggregation this field belongs to is already known from the Redis field
    name (``agg.window.tile_index`` decomposition happens in ``online.keys``), and
    checking the decoded length against the expected arity catches a cross-wired read,
    for example a caller asking for an AVG's two doubles from a SUM's one, as a decode
    error rather than as a silently short tuple.
    """
    body = _body_format(arity)
    expected_length = _HEADER.size + body.size
    if len(payload) != expected_length:
        raise CodecError(f"expected {expected_length} bytes for arity {arity}, got {len(payload)}")
    (version,) = _HEADER.unpack_from(payload, 0)
    if version != CODEC_VERSION:
        raise CodecError(
            f"unknown codec version {version}; this reader supports version {CODEC_VERSION} "
            f"only, refusing rather than risking a misread"
        )
    return body.unpack_from(payload, _HEADER.size)
