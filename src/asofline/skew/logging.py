"""What the serving layer logs, and the one validation that protects the detector.

A skew detector is only as trustworthy as its own input. If the serving layer could log a
feature vector under the wrong shape, an entity-key typo, or a stale view version without
anything noticing, an apparent "skew" finding could just as easily be a logging bug, and
the detector would have no way to tell the difference. ``build_log_entry`` is the one
place a served vector becomes a log entry, and it refuses to build one whose feature names
do not exactly match what the view declares.

The wire shape is one nested JSON object per served request, matching
``asofline.demo.events.EngagementEvent``'s own convention: a single Kafka message the
serving layer can fire and forget, with no per-feature fan-out on the request path.
Flattening it into the long, per-feature rows that ``offline.tables.serving_log_ddl``
stores is a downstream ingestion concern, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asofline.definitions.view import FeatureView


class FeatureLogError(ValueError):
    """A feature vector does not match its view's declared shape closely enough to log."""


@dataclass(frozen=True, slots=True)
class FeatureLogEntry:
    """One served feature vector, ready to serialize onto the ``feature_logs`` topic."""

    log_id: str
    view_name: str
    view_version: int
    entity_keys: dict[str, str]
    request_ts_ms: int
    served_at_ms: int
    features: dict[str, float | None]

    def __post_init__(self) -> None:
        if self.served_at_ms < self.request_ts_ms:
            raise FeatureLogError(
                f"{self.log_id}: served_at_ms ({self.served_at_ms}) precedes "
                f"request_ts_ms ({self.request_ts_ms}); a response cannot be served "
                f"before the instant it was asked to be served as of"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "view_name": self.view_name,
            "view_version": self.view_version,
            "entity_keys": dict(self.entity_keys),
            "request_ts_ms": self.request_ts_ms,
            "served_at_ms": self.served_at_ms,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureLogEntry:
        return cls(
            log_id=str(payload["log_id"]),
            view_name=str(payload["view_name"]),
            view_version=int(payload["view_version"]),
            entity_keys={str(k): str(v) for k, v in payload["entity_keys"].items()},
            request_ts_ms=int(payload["request_ts_ms"]),
            served_at_ms=int(payload["served_at_ms"]),
            features={
                str(k): (None if v is None else float(v)) for k, v in payload["features"].items()
            },
        )


def build_log_entry(
    view: FeatureView,
    entity_keys: dict[str, str],
    *,
    log_id: str,
    request_ts_ms: int,
    served_at_ms: int,
    features: dict[str, float | None],
) -> FeatureLogEntry:
    """Build a log entry, refusing one whose feature set does not match the view exactly.

    Exact set equality rather than a subset check in either direction: a missing feature
    would silently drop a column the detector expects to find every request, and an extra
    one would mean the serving layer computed something outside its own contract. Both are
    bugs worth failing loudly on rather than logging past.
    """
    declared = set(view.feature_names)
    got = set(features)
    if got != declared:
        missing = declared - got
        extra = got - declared
        raise FeatureLogError(
            f"{view.name}: logged features do not match the view's declared features "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    missing_keys = [key for key in view.join_keys if key not in entity_keys]
    if missing_keys:
        raise FeatureLogError(f"{view.name}: missing entity key value(s) {missing_keys}")

    return FeatureLogEntry(
        log_id=log_id,
        view_name=view.name,
        view_version=view.version,
        entity_keys={key: entity_keys[key] for key in view.join_keys},
        request_ts_ms=request_ts_ms,
        served_at_ms=served_at_ms,
        features=dict(features),
    )
