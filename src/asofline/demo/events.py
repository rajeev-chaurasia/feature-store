"""The raw event this project ingests.

Times are epoch **milliseconds as ints** everywhere below the boundary layers. Datetimes
are converted at the edges only. Two reasons: tile indices are integer divisions of a
timestamp and floats make that subtly wrong near boundaries, and a naive datetime that
picks up a local timezone somewhere in a Spark round trip is a class of bug that is very
hard to see and very easy to avoid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    IMPRESSION = "impression"
    WATCH = "watch"
    LIKE = "like"
    SHARE = "share"
    FOLLOW = "follow"


def to_millis(moment: datetime) -> int:
    if moment.tzinfo is None:
        raise ValueError(f"refusing a naive datetime: {moment!r}")
    return int(moment.timestamp() * 1000)


def from_millis(millis: int) -> datetime:
    return datetime.fromtimestamp(millis / 1000, tz=UTC)


@dataclass(frozen=True, slots=True)
class EngagementEvent:
    """One interaction.

    ``event_ts`` is when it happened. ``created_ts`` is when this system first saw it.
    ``created_ts >= event_ts`` always, and the gap is the late tail that the whole
    point-in-time argument turns on.
    """

    event_id: str
    event_type: EventType
    user_id: str
    video_id: str
    event_ts: int
    created_ts: int
    watch_seconds: float = 0.0
    liked: int = 0
    shared: int = 0

    def __post_init__(self) -> None:
        if self.created_ts < self.event_ts:
            raise ValueError(
                f"event {self.event_id} was created {self.event_ts - self.created_ts}ms "
                f"before it happened"
            )

    @property
    def lateness_ms(self) -> int:
        return self.created_ts - self.event_ts

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_type"] = str(self.event_type)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EngagementEvent:
        return cls(
            event_id=str(payload["event_id"]),
            event_type=EventType(payload["event_type"]),
            user_id=str(payload["user_id"]),
            video_id=str(payload["video_id"]),
            event_ts=int(payload["event_ts"]),
            created_ts=int(payload["created_ts"]),
            watch_seconds=float(payload.get("watch_seconds", 0.0)),
            liked=int(payload.get("liked", 0)),
            shared=int(payload.get("shared", 0)),
        )
