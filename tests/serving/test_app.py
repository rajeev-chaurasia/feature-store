"""The FastAPI serving app, exercised end to end against a real (test) Redis.

This only checks the HTTP plumbing: request/response shape, error mapping, health check.
The actual read logic is ``online.store.OnlineStore``, already covered by
``tests/integration/test_online_store.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from asofline.demo.views import USER_ENGAGEMENT
from asofline.online.store import OnlineStore
from asofline.serving.app import app
from tests.redis_fixtures import TEST_REDIS_URL, flush_test_database, make_test_redis, seed_tile

pytestmark = pytest.mark.integration

FIVE_MIN = 5 * 60_000
# Same fixed, deliberately off-boundary instant used in tests/integration/test_online_store.
AS_OF_MS = 1_767_225_600_000 + 12 * 60_000


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        # The app's lifespan built a store from SETTINGS.redis_url on startup. Replace it
        # with one pointed at the database reserved for this test suite, so test data
        # cannot collide with a real deployment or another agent's own test run on the
        # default database.
        test_store = OnlineStore.from_url(TEST_REDIS_URL)
        app.state.store = test_store
        try:
            yield test_client
        finally:
            # TestClient drives the app through its own event loop (a portal running in a
            # worker thread), so the store's connections belong to that loop, not
            # whichever loop a plain asyncio.run() here would create. Close it through the
            # same portal to avoid an "attached to a different loop" error.
            test_client.portal.call(test_store.close)


def _seed_one_tile(
    *, user_id: str, agg_name: str, granularity_ms: int, event_ts_ms: int, state: tuple[float, ...]
) -> None:
    async def _do() -> None:
        redis = make_test_redis()
        await flush_test_database(redis)
        await seed_tile(
            redis,
            USER_ENGAGEMENT,
            {"user_id": user_id},
            agg_name=agg_name,
            granularity_ms=granularity_ms,
            event_ts_ms=event_ts_ms,
            state=state,
        )
        await redis.aclose()

    asyncio.run(_do())


class TestGetOnlineFeatures:
    def test_valid_request_returns_200_with_expected_shape_and_values(
        self, client: TestClient
    ) -> None:
        _seed_one_tile(
            user_id="u-app-1",
            agg_name="watch_seconds_sum",
            granularity_ms=FIVE_MIN,
            event_ts_ms=AS_OF_MS - 10 * 60_000,
            state=(12.0,),
        )

        response = client.post(
            "/get-online-features",
            json={
                "view": "user_engagement",
                "entities": [{"user_id": "u-app-1"}],
                "as_of_ms": AS_OF_MS,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["as_of_ms"] == AS_OF_MS
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["user_id"] == "u-app-1"
        assert result["features"]["watch_seconds_sum_1h"] == pytest.approx(12.0)
        assert set(result["features"]) == set(USER_ENGAGEMENT.feature_names)

    def test_unknown_view_name_returns_a_4xx_with_a_clear_error(self, client: TestClient) -> None:
        response = client.post(
            "/get-online-features",
            json={"view": "not_a_real_view", "entities": [{"user_id": "u1"}]},
        )
        assert 400 <= response.status_code < 500
        assert "not_a_real_view" in response.json()["detail"]

    def test_as_of_ms_defaults_to_now_when_omitted(self, client: TestClient) -> None:
        response = client.post(
            "/get-online-features",
            json={"view": "user_engagement", "entities": [{"user_id": "u-app-never-seen"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["as_of_ms"] > 0
        assert all(value is None for value in body["results"][0]["features"].values())


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
