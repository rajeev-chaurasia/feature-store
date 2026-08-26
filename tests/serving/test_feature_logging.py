"""Does a served request actually produce a consumable entry on ``feature_logs``?

This is the one place the fire-and-forget path is checked against a real broker rather
than mocked: everything upstream of the Kafka publish (sampling, building the entry,
scheduling the background task) could be individually correct and still not add up to a
message actually reaching the topic P5's ingestion job reads from.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

import pytest
from confluent_kafka import Consumer, TopicPartition
from fastapi.testclient import TestClient

from asofline.config import SETTINGS
from asofline.demo.views import USER_ENGAGEMENT
from asofline.online.store import OnlineStore
from asofline.serving.app import app
from asofline.skew.logging import FeatureLogEntry
from tests.redis_fixtures import TEST_REDIS_URL
from tests.serving.test_app import _seed_one_tile

pytestmark = pytest.mark.integration

FIVE_MIN = 5 * 60_000
AS_OF_MS = 1_767_225_600_000 + 12 * 60_000


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        test_store = OnlineStore.from_url(TEST_REDIS_URL)
        app.state.store = test_store
        app.state.log_failures = 0
        try:
            yield test_client
        finally:
            test_client.portal.call(test_store.close)


def _drain_feature_logs(*, timeout_s: float = 10.0) -> list[FeatureLogEntry]:
    """Read whatever is currently on ``feature_logs``, from the beginning.

    A fresh, uniquely named consumer group every call: the topic is shared with whatever
    else might be produced to it, and a fresh group with ``earliest`` guarantees this test
    sees every message ever published rather than only what arrives after it starts
    polling, which is what a filter on entity id (done by the caller) needs to work.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": SETTINGS.kafka_bootstrap,
            "group.id": f"test-feature-logs-{uuid.uuid4().hex}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    try:
        metadata = consumer.list_topics(SETTINGS.feature_log_topic, timeout=10.0)
        partitions = [
            TopicPartition(SETTINGS.feature_log_topic, p)
            for p in metadata.topics[SETTINGS.feature_log_topic].partitions
        ]
        consumer.assign(partitions)
        # Poll for the whole window unconditionally rather than breaking early on a
        # run of empty polls: a freshly created consumer group's first few polls can
        # return None while group/metadata setup completes even though messages already
        # sit at the earliest offset, and an early-break heuristic here previously cut
        # the loop before that setup finished, missing a message that later manual
        # inspection proved had already been delivered.
        entries: list[FeatureLogEntry] = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = consumer.poll(1.0)
            if message is None or message.error():
                continue
            entries.append(FeatureLogEntry.from_dict(json.loads(message.value())))
        return entries
    finally:
        consumer.close()


class TestFeatureLoggingReachesKafka:
    def test_a_served_request_publishes_a_matching_log_entry(self, client: TestClient) -> None:
        user_id = f"u_log_{uuid.uuid4().hex[:12]}"
        _seed_one_tile(
            user_id=user_id,
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 60_000,
            state=(42.0,),
        )

        response = client.post(
            "/get-online-features",
            json={
                "view": "user_engagement",
                "entities": [{"user_id": user_id}],
                "as_of_ms": AS_OF_MS,
            },
        )
        assert response.status_code == 200
        served_features = response.json()["results"][0]["features"]

        # Give the background task, scheduled on the TestClient's own portal loop, a
        # chance to run and the producer a chance to flush before polling Kafka for it.
        time.sleep(0.5)
        app.state.producer.flush(5.0)

        entries = _drain_feature_logs()
        matches = [e for e in entries if e.entity_keys.get("user_id") == user_id]
        assert len(matches) == 1, (
            f"expected exactly one log entry for {user_id}, got {len(matches)}"
        )

        entry = matches[0]
        assert entry.view_name == "user_engagement"
        assert entry.view_version == USER_ENGAGEMENT.version
        assert entry.request_ts_ms == AS_OF_MS
        assert entry.features == served_features
        assert app.state.log_failures == 0

    def test_an_unknown_view_never_reaches_the_logger(self, client: TestClient) -> None:
        """A 404 must not produce a log entry: there is no served vector to log."""
        response = client.post(
            "/get-online-features", json={"view": "not_a_real_view", "entities": [{"user_id": "x"}]}
        )
        assert response.status_code == 404
        assert app.state.log_failures == 0
