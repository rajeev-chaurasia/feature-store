"""The one validation protecting the P5 skew detector from trusting a malformed log entry."""

from __future__ import annotations

import pytest

from asofline.demo.views import USER_ENGAGEMENT
from asofline.skew.logging import FeatureLogEntry, FeatureLogError, build_log_entry


def _valid_features() -> dict[str, float | None]:
    return dict.fromkeys(USER_ENGAGEMENT.feature_names, 1.0)


class TestBuildLogEntry:
    def test_a_correctly_shaped_vector_builds(self) -> None:
        entry = build_log_entry(
            USER_ENGAGEMENT,
            {"user_id": "u1"},
            log_id="log-1",
            request_ts_ms=1_000,
            served_at_ms=1_050,
            features=_valid_features(),
        )
        assert entry.view_name == "user_engagement"
        assert entry.view_version == USER_ENGAGEMENT.version
        assert entry.entity_keys == {"user_id": "u1"}

    def test_a_missing_feature_is_rejected(self) -> None:
        features = _valid_features()
        del features[USER_ENGAGEMENT.feature_names[0]]
        with pytest.raises(FeatureLogError, match="missing="):
            build_log_entry(
                USER_ENGAGEMENT,
                {"user_id": "u1"},
                log_id="log-1",
                request_ts_ms=1_000,
                served_at_ms=1_050,
                features=features,
            )

    def test_an_extra_feature_is_rejected(self) -> None:
        features = {**_valid_features(), "not_a_real_feature": 1.0}
        with pytest.raises(FeatureLogError, match="extra="):
            build_log_entry(
                USER_ENGAGEMENT,
                {"user_id": "u1"},
                log_id="log-1",
                request_ts_ms=1_000,
                served_at_ms=1_050,
                features=features,
            )

    def test_a_missing_entity_key_is_rejected(self) -> None:
        with pytest.raises(FeatureLogError, match="user_id"):
            build_log_entry(
                USER_ENGAGEMENT,
                {},
                log_id="log-1",
                request_ts_ms=1_000,
                served_at_ms=1_050,
                features=_valid_features(),
            )

    def test_extra_entity_keys_are_dropped_not_stored(self) -> None:
        """Only the view's declared join keys are kept, in the view's declared order.

        A caller's dict may carry unrelated request context; storing it verbatim would
        make the logged entity_keys shape depend on caller hygiene rather than on the
        view's contract.
        """
        entry = build_log_entry(
            USER_ENGAGEMENT,
            {"user_id": "u1", "unrelated": "x"},
            log_id="log-1",
            request_ts_ms=1_000,
            served_at_ms=1_050,
            features=_valid_features(),
        )
        assert entry.entity_keys == {"user_id": "u1"}

    def test_null_feature_values_are_allowed(self) -> None:
        """A stale or unseen entity is served nulls, and logging must accept that as is."""
        features = dict.fromkeys(USER_ENGAGEMENT.feature_names, None)
        entry = build_log_entry(
            USER_ENGAGEMENT,
            {"user_id": "u1"},
            log_id="log-1",
            request_ts_ms=1_000,
            served_at_ms=1_050,
            features=features,
        )
        assert all(value is None for value in entry.features.values())


class TestFeatureLogEntry:
    def test_served_before_requested_is_rejected(self) -> None:
        with pytest.raises(FeatureLogError, match="precedes"):
            FeatureLogEntry(
                log_id="log-1",
                view_name="user_engagement",
                view_version=1,
                entity_keys={"user_id": "u1"},
                request_ts_ms=1_000,
                served_at_ms=999,
                features={},
            )

    def test_round_trip_through_the_wire_form(self) -> None:
        entry = build_log_entry(
            USER_ENGAGEMENT,
            {"user_id": "u1"},
            log_id="log-1",
            request_ts_ms=1_000,
            served_at_ms=1_050,
            features={**_valid_features(), "count_1h": None},
        )
        assert FeatureLogEntry.from_dict(entry.to_dict()) == entry
