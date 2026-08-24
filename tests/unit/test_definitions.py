"""Definition validation. Every rule here fires at construction time, on purpose."""

from __future__ import annotations

from datetime import timedelta

import pytest

from asofline.definitions import (
    AggFunction,
    Aggregation,
    DefinitionError,
    Entity,
    EventSource,
    FeatureView,
    Registry,
    format_window,
)
from asofline.demo.views import DEMO_REGISTRY, ENGAGEMENT_SOURCE, USER

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
WEEK = timedelta(days=7)


def _view(**overrides: object) -> FeatureView:
    defaults: dict[str, object] = {
        "name": "probe",
        "entities": (USER,),
        "aggregations": (Aggregation(AggFunction.SUM, (HOUR,), column="watch_seconds"),),
        "source": ENGAGEMENT_SOURCE,
        "ttl": DAY,
    }
    return FeatureView(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestWindowLabels:
    @pytest.mark.parametrize(
        ("window", "label"),
        [(HOUR, "1h"), (DAY, "1d"), (WEEK, "7d"), (timedelta(minutes=5), "5m")],
    )
    def test_label(self, window: timedelta, label: str) -> None:
        assert format_window(window) == label

    def test_labels_prefer_the_coarsest_exact_unit(self) -> None:
        assert format_window(timedelta(hours=48)) == "2d"
        assert format_window(timedelta(minutes=90)) == "90m"


class TestAggregation:
    def test_count_must_not_name_a_column(self) -> None:
        with pytest.raises(DefinitionError, match="must not name a source column"):
            Aggregation(AggFunction.COUNT, (HOUR,), column="watch_seconds")

    def test_sum_must_name_a_column(self) -> None:
        with pytest.raises(DefinitionError, match="needs a source column"):
            Aggregation(AggFunction.SUM, (HOUR,))

    def test_repeated_window_is_rejected(self) -> None:
        with pytest.raises(DefinitionError, match="repeats a window"):
            Aggregation(AggFunction.SUM, (HOUR, HOUR), column="watch_seconds")

    def test_no_windows_is_rejected(self) -> None:
        with pytest.raises(DefinitionError, match="declares no windows"):
            Aggregation(AggFunction.SUM, (), column="watch_seconds")

    def test_feature_names(self) -> None:
        aggregation = Aggregation(AggFunction.SUM, (HOUR, DAY), column="watch_seconds")
        assert aggregation.feature_names == ("watch_seconds_sum_1h", "watch_seconds_sum_1d")
        assert Aggregation(AggFunction.COUNT, (HOUR,)).feature_names == ("count_1h",)

    def test_the_algebra_split_is_what_the_batch_compiler_branches_on(self) -> None:
        assert AggFunction.SUM.has_inverse
        assert AggFunction.COUNT.has_inverse
        assert AggFunction.AVG.has_inverse
        assert not AggFunction.MIN.has_inverse
        assert not AggFunction.MAX.has_inverse


class TestFeatureView:
    def test_ttl_shorter_than_the_longest_window_is_rejected(self) -> None:
        with pytest.raises(DefinitionError, match="shorter than the longest window"):
            _view(
                aggregations=(Aggregation(AggFunction.SUM, (WEEK,), column="watch_seconds"),),
                ttl=DAY,
            )

    def test_duplicate_feature_within_a_view_is_rejected(self) -> None:
        duplicate = Aggregation(AggFunction.SUM, (HOUR,), column="watch_seconds")
        with pytest.raises(DefinitionError, match="duplicate feature"):
            _view(aggregations=(duplicate, duplicate))

    def test_repeated_join_key_is_rejected(self) -> None:
        twin = Entity(name="viewer", join_key="user_id")
        with pytest.raises(DefinitionError, match="repeated join key"):
            _view(entities=(USER, twin))

    def test_a_view_needs_an_entity_and_an_aggregation(self) -> None:
        with pytest.raises(DefinitionError, match="at least one entity"):
            _view(entities=())
        with pytest.raises(DefinitionError, match="at least one aggregation"):
            _view(aggregations=())

    def test_name_must_be_lower_snake_case(self) -> None:
        # The name becomes an Iceberg table name and a Redis key segment. Rejecting here
        # is cheaper than discovering the disagreement in whichever layer is strictest.
        with pytest.raises(DefinitionError, match="lower snake case"):
            _view(name="UserEngagement")

    def test_grids_report_the_per_event_write_fan_out(self) -> None:
        view = _view(
            aggregations=(Aggregation(AggFunction.SUM, (HOUR, WEEK), column="watch_seconds"),),
            ttl=WEEK,
        )
        assert view.grids == (timedelta(minutes=5), HOUR)
        assert view.retention_for(timedelta(minutes=5)) == HOUR
        assert view.retention_for(HOUR) == WEEK

    def test_qualified_names_carry_the_view(self) -> None:
        assert _view().qualified_feature_names == ("probe:watch_seconds_sum_1h",)


class TestEventSource:
    def test_collapsing_the_two_timestamps_is_rejected(self) -> None:
        with pytest.raises(DefinitionError, match="must be distinct columns"):
            EventSource(
                topic="t", raw_table="t", timestamp_field="ts", created_timestamp_field="ts"
            )


class TestRegistry:
    def test_demo_registry_is_valid_and_covers_both_halves_of_the_algebra(self) -> None:
        functions = {
            aggregation.function
            for view in DEMO_REGISTRY.views
            for aggregation in view.aggregations
        }
        assert any(f.has_inverse for f in functions)
        assert any(not f.has_inverse for f in functions), "no range-path feature to keep it honest"

    def test_demo_registry_spans_both_resolution_tiers(self) -> None:
        grids = {grid for view in DEMO_REGISTRY.views for grid in view.grids}
        assert len(grids) == 2

    def test_same_feature_in_two_views_is_fine_once_qualified(self) -> None:
        """Both demo views aggregate watch_seconds the same way, and that is legal.

        The short name collides; the qualified name does not. This is the case that made
        qualification necessary in the first place.
        """
        short_names = [name for view in DEMO_REGISTRY.views for name in view.feature_names]
        assert len(short_names) != len(set(short_names)), "the collision this test guards is gone"
        assert len(DEMO_REGISTRY.feature_names) == len(set(DEMO_REGISTRY.feature_names))

    def test_two_versions_of_one_view_collide_on_qualified_names(self) -> None:
        view = _view()
        with pytest.raises(DefinitionError, match="flat map"):
            Registry(views=(view, _view(version=2)))

    def test_inconsistent_entity_definition_is_rejected(self) -> None:
        other = Entity(name="user", join_key="uid")
        with pytest.raises(DefinitionError, match="defined inconsistently"):
            Registry(views=(_view(), _view(name="probe_two", entities=(other,))))

    def test_view_for_feature_requires_qualification(self) -> None:
        with pytest.raises(DefinitionError, match="not qualified"):
            DEMO_REGISTRY.view_for_feature("count_1h")

    def test_view_for_feature_rejects_a_bad_suffix(self) -> None:
        with pytest.raises(DefinitionError, match="exports no feature"):
            DEMO_REGISTRY.view_for_feature("user_engagement:count_9h")

    def test_view_for_feature_resolves(self) -> None:
        view = DEMO_REGISTRY.view_for_feature("video_engagement:watch_seconds_max_1d")
        assert view.name == "video_engagement"
