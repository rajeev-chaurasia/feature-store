"""Where raw events come from, and which of their timestamps means what.

The two timestamp fields are the whole point of this module. ``timestamp_field`` is when
the thing happened. ``created_timestamp_field`` is when this system first knew about it.
They differ whenever an event arrives late, and the gap between them is what separates a
point-in-time correct backfill from a leaky one.
"""

from __future__ import annotations

from dataclasses import dataclass

from asofline.definitions.errors import DefinitionError
from asofline.definitions.naming import validate_identifier


@dataclass(frozen=True, slots=True)
class EventSource:
    """A Kafka topic and the Iceberg table its events land in.

    One source feeds both paths: the streaming consumer reads the topic, the batch jobs
    read the table, and both are told the same two timestamp column names.
    """

    topic: str
    raw_table: str
    timestamp_field: str = "event_ts"
    created_timestamp_field: str = "created_ts"

    def __post_init__(self) -> None:
        if not self.topic:
            raise DefinitionError("event source needs a topic")
        validate_identifier(self.raw_table, kind="raw table")
        validate_identifier(self.timestamp_field, kind="timestamp field")
        validate_identifier(self.created_timestamp_field, kind="created timestamp field")
        if self.timestamp_field == self.created_timestamp_field:
            raise DefinitionError(
                "event time and created time must be distinct columns; collapsing them "
                "makes late arrival invisible and every backfill silently leaky"
            )
