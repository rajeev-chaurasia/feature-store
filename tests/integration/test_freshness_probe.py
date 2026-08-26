"""P4's freshness probe, exercised against the real stack at a small scale.

Full scale (100+ iterations) lives in ``scripts/run_freshness_probe.py``; this suite runs
a handful of probes so it stays fast enough for the normal integration run while still
exercising the real mechanism: a genuinely separate consumer subprocess, a real Kafka
publish, and real polling of the online store.

Targets ``tests.redis_fixtures.TEST_REDIS_URL`` (database 15), the index this project's
test suites already set aside so a probe's writes cannot collide with a production
consumer's default database or with data another suite has live at the same moment. The
Kafka topic itself is the shared ``engagement_events`` topic (there is no test-only
topic), which is safe here because every probe consumer group is a fresh, run-scoped name
(``asofline.bench.freshness.run_freshness_probe_with_consumer``) and every probe event
carries a fresh, never-reused ``user_id``: nothing this suite writes can be observed by,
or interfere with, anything else reading that topic.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from confluent_kafka import KafkaException, Producer

from asofline.artifacts import (
    ARTIFACT_KIND_STREAMING_LAG,
    artifact_validation_errors,
    build_artifact,
)
from asofline.bench.freshness import (
    DEFAULT_FEATURE_NAME,
    FreshnessProbeConfig,
    _poll_until_visible,
    _publish,
    run_freshness_probe_with_consumer,
)
from asofline.config import SETTINGS
from asofline.demo.events import EngagementEvent, EventType
from asofline.demo.views import USER_ENGAGEMENT
from asofline.online.store import OnlineStore
from tests.redis_fixtures import TEST_REDIS_URL, flush_test_database, make_test_redis, run_async

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_kafka() -> Iterator[None]:
    """Skip this whole module if the broker is not reachable, rather than hanging."""
    producer = Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})
    try:
        producer.list_topics(timeout=5.0)
    except KafkaException as error:
        pytest.skip(f"kafka not reachable at {SETTINGS.kafka_bootstrap}: {error}")
    yield


def _config(**overrides: object) -> FreshnessProbeConfig:
    base = {
        "iterations": 5,
        "kafka_bootstrap": SETTINGS.kafka_bootstrap,
        "events_topic": SETTINGS.events_topic,
        "redis_url": TEST_REDIS_URL,
        "poll_interval_s": 0.03,
        "probe_timeout_s": 10.0,
        "warmup_timeout_s": 45.0,
    }
    base.update(overrides)
    return FreshnessProbeConfig(**base)  # type: ignore[arg-type]


def test_probe_events_are_detected_within_a_generous_timeout_when_consumer_is_running() -> None:
    """The happy path: a real consumer subprocess, real Kafka publishes, real polling.

    Five samples, not a fixed pass/fail on one, so a single slow poll cycle cannot make
    this test flaky: every probe must succeed and every latency must be a small positive
    number, but no assumption is made about which poll cycle it lands on.
    """
    config = _config(iterations=5)
    result = run_freshness_probe_with_consumer(
        config, view=USER_ENGAGEMENT, feature_name=DEFAULT_FEATURE_NAME
    )

    assert result.failure_count == 0, "every probe should become visible while a consumer runs"
    assert result.completed_count == 5
    assert result.requested_count == 5
    for sample_ms in result.raw_samples_ms:
        assert 0.0 <= sample_ms < config.probe_timeout_s * 1000


@run_async
async def test_timeout_path_is_exercised_when_no_consumer_is_running() -> None:
    """The failure path, demonstrated rather than only assumed to work.

    No consumer subprocess is started here: a probe event is published straight onto
    Kafka, but with nothing consuming it into Redis, the online store can never reflect
    its contribution. ``_poll_until_visible`` must return ``None`` once its (short)
    timeout elapses, not hang and not fabricate a latency.
    """
    redis = make_test_redis()
    await flush_test_database(redis)
    store = OnlineStore(redis)
    producer = Producer({"bootstrap.servers": SETTINGS.kafka_bootstrap})
    try:
        event = EngagementEvent(
            event_id="freshness-timeout-probe",
            event_type=EventType.WATCH,
            user_id="fresh-timeout-never-arrives",
            video_id="v-fresh-timeout-never-arrives",
            event_ts=1_800_000_000_000,
            created_ts=1_800_000_000_000,
            watch_seconds=123.456,
        )
        _publish(producer, SETTINGS.events_topic, event, flush_timeout_s=5.0)

        observed_at = await _poll_until_visible(
            store,
            USER_ENGAGEMENT,
            {"user_id": event.user_id},
            DEFAULT_FEATURE_NAME,
            event.watch_seconds,
            timeout_s=0.3,
            poll_interval_s=0.05,
        )

        assert observed_at is None
    finally:
        await flush_test_database(redis)
        await store.close()


def test_artifact_built_from_probe_samples_passes_validation() -> None:
    """The artifact this project would actually commit must pass its own validator.

    Reuses the happy-path run rather than re-running the probe a third time: what matters
    here is that ``build_artifact`` applied to real, freshly measured samples produces
    something ``artifact_validation_errors`` accepts with zero complaints, exactly the
    contract ``scripts/run_freshness_probe.py`` depends on.
    """
    config = _config(iterations=5)
    result = run_freshness_probe_with_consumer(
        config, view=USER_ENGAGEMENT, feature_name=DEFAULT_FEATURE_NAME
    )
    assert result.raw_samples_ms, "need at least one real sample to build an artifact from"

    artifact = build_artifact(
        artifact_kind=ARTIFACT_KIND_STREAMING_LAG,
        created_at="2026-08-25T00:00:00+00:00",
        config={
            "iterations": config.iterations,
            "feature_name": DEFAULT_FEATURE_NAME,
            "view": USER_ENGAGEMENT.name,
            "requested_count": result.requested_count,
            "failure_count": result.failure_count,
        },
        raw_samples_ms=result.raw_samples_ms,
    )

    assert artifact_validation_errors(artifact) == []
