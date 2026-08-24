"""The demo registry.

Deliberately exercises both halves of the aggregation algebra. ``sum``, ``count`` and
``avg`` have inverses and take the prefix-sum path in the batch compiler; ``max`` does
not and takes the range path. A registry with only summable features would let a broken
range path ship unnoticed.

It also spans both resolution tiers: 1h and 24h land on the 5-minute grid, 7d on the
1-hour grid, so the per-event write fan-out of two grids is exercised from the start.
"""

from __future__ import annotations

from datetime import timedelta

from asofline.definitions import (
    AggFunction,
    Aggregation,
    Entity,
    EventSource,
    FeatureView,
    Registry,
)

ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)
SEVEN_DAYS = timedelta(days=7)

USER = Entity(name="user", join_key="user_id", description="A viewer.")
VIDEO = Entity(name="video", join_key="video_id", description="A short video.")

ENGAGEMENT_SOURCE = EventSource(
    topic="engagement_events",
    raw_table="engagement_events",
    timestamp_field="event_ts",
    created_timestamp_field="created_ts",
)

USER_ENGAGEMENT = FeatureView(
    name="user_engagement",
    entities=(USER,),
    source=ENGAGEMENT_SOURCE,
    ttl=SEVEN_DAYS,
    aggregations=(
        Aggregation(AggFunction.SUM, (ONE_HOUR, ONE_DAY, SEVEN_DAYS), column="watch_seconds"),
        Aggregation(AggFunction.COUNT, (ONE_HOUR, ONE_DAY)),
        Aggregation(AggFunction.SUM, (ONE_DAY, SEVEN_DAYS), column="liked"),
        Aggregation(AggFunction.AVG, (ONE_DAY,), column="watch_seconds"),
    ),
    description="What one viewer has been doing.",
)

VIDEO_ENGAGEMENT = FeatureView(
    name="video_engagement",
    entities=(VIDEO,),
    source=ENGAGEMENT_SOURCE,
    ttl=ONE_DAY,
    aggregations=(
        Aggregation(AggFunction.SUM, (ONE_HOUR, ONE_DAY), column="watch_seconds"),
        Aggregation(AggFunction.SUM, (ONE_HOUR, ONE_DAY), column="shared"),
        # No inverse, so this is the feature that keeps the range path honest.
        Aggregation(AggFunction.MAX, (ONE_DAY,), column="watch_seconds"),
    ),
    description="How one video is performing.",
)

DEMO_REGISTRY = Registry(views=(USER_ENGAGEMENT, VIDEO_ENGAGEMENT))
