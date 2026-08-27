"""Run the P3 done-test for real: a committed p50/p99 at a stated QPS, against a live server.

This is not a description of how one would benchmark the online store. It seeds real
Redis state through the actual Kafka-to-Redis write path (``streaming.consumer``), starts
the actual FastAPI app in a subprocess, drives it with the actual open-loop load generator,
and writes a committed artifact through ``asofline.artifacts`` with the full raw sample
array, not only a summary. Every number this script prints is recomputed by
``scripts/validate_artifacts.py`` from the committed file, not trusted from this process's
own memory.

Each configuration is run 3 times, not once. A single trial pair does not distinguish a
real effect from ordinary run-to-run variance on a shared development machine, and an
earlier version of this script that ran once per configuration reported a p99 gap that
this rerun shows was mostly noise: p90 and p99 vary more between repeated trials of the
same configuration than the two configurations differ from each other.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import redis as sync_redis

from asofline.artifacts import (
    ARTIFACT_KIND_ONLINE_LATENCY,
    artifact_validation_errors,
    build_artifact,
    write_artifact_atomic,
)
from asofline.bench.load import LoadTestConfig, run_load_test, sample_entity_pool
from asofline.config import SETTINGS
from asofline.demo.events import EngagementEvent, EventType
from asofline.demo.views import DEMO_REGISTRY, USER_ENGAGEMENT
from asofline.streaming.consumer import process_event

RUN_ID = "2026-08-24-online-latency"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / RUN_ID
PORT = 8931


def _port_for(sample_rate: float) -> int:
    return PORT if sample_rate > 0 else PORT + 1


def seed_population(config: LoadTestConfig) -> None:
    """Populate Redis for exactly the entity ids the load generator will query.

    A benchmark against an empty store measures the cost of the null path, not of the
    rollup, and every entity the generator can draw has to actually carry data for the
    p50/p99 to mean anything.
    """
    client = sync_redis.Redis.from_url(SETTINGS.redis_url)
    pool = sample_entity_pool(config, "user_id")
    now_ms = int(time.time() * 1000)

    for index, entity in enumerate(pool):
        user_id = entity["user_id"]
        # A handful of events per entity, spread over the last few hours, so every grid a
        # window touches has at least one populated tile and the head is non-trivial.
        for offset_minutes, watch_seconds in ((5, 12.0), (40, 8.0), (130, 30.0)):
            event = EngagementEvent(
                event_id=f"seed-{index}-{offset_minutes}",
                event_type=EventType.WATCH,
                user_id=user_id,
                video_id=f"v{index % 500:04d}",
                event_ts=now_ms - offset_minutes * 60_000,
                created_ts=now_ms - offset_minutes * 60_000,
                watch_seconds=watch_seconds,
            )
            process_event(client, event, DEMO_REGISTRY)
    client.close()
    print(f"seeded {len(pool)} entities")


def start_server(*, port: int, feature_log_sample_rate: float) -> subprocess.Popen[bytes]:
    import os as _os

    env = {**_os.environ, "ASOFLINE_FEATURE_LOG_SAMPLE_RATE": str(feature_log_sample_rate)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "asofline.serving.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if response.status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    process.terminate()
    raise RuntimeError("serving app did not become healthy in time")


def run_one(*, feature_log_sample_rate: float, out_name: str, trial: int) -> dict[str, float | int]:
    port = _port_for(feature_log_sample_rate)
    config = LoadTestConfig(
        base_url=f"http://127.0.0.1:{port}",
        view=USER_ENGAGEMENT.name,
        qps=50.0,
        duration_s=20.0,
        entity_pool_size=200,
        seed=20260823,
    )
    seed_population(config)

    server = start_server(port=port, feature_log_sample_rate=feature_log_sample_rate)
    try:

        async def run() -> tuple[list[float], int, int]:
            async with httpx.AsyncClient(base_url=config.base_url) as client:
                result = await run_load_test(client, config, "user_id")
            return result.raw_samples_ms, result.error_count, result.requested_count

        raw_samples_ms, error_count, requested_count = asyncio.run(run())
    finally:
        server.terminate()
        server.wait(timeout=10)

    print(
        f"[sample_rate={feature_log_sample_rate}] requested={requested_count} "
        f"completed={len(raw_samples_ms)} errors={error_count}"
    )
    if error_count:
        print(f"WARNING: {error_count} requests failed and are excluded from raw_samples_ms")

    artifact = build_artifact(
        artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
        created_at=datetime.now(UTC).isoformat(),
        config={
            "view": config.view,
            "qps": config.qps,
            "duration_s": config.duration_s,
            "entity_pool_size": config.entity_pool_size,
            "seed": config.seed,
            "requested_count": requested_count,
            "error_count": error_count,
            "feature_log_sample_rate": feature_log_sample_rate,
            "trial": trial,
        },
        raw_samples_ms=raw_samples_ms,
    )

    errors = artifact_validation_errors(artifact)
    if errors:
        raise RuntimeError(f"artifact failed validation: {errors}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / out_name
    write_artifact_atomic(artifact, out_path)

    stats = artifact["statistics"]
    print(
        f"  -> {out_path.name}: p50={stats['median']:.2f}ms p90={stats['p90']:.2f}ms "
        f"p99={stats['p99']:.2f}ms max={stats['max']:.2f}ms n={stats['count']}"
    )
    return stats


def main() -> int:
    trials = 3
    with_logging = [
        run_one(
            feature_log_sample_rate=1.0,
            out_name=f"online_latency_trial{trial}.json",
            trial=trial,
        )
        for trial in range(1, trials + 1)
    ]
    without_logging = [
        run_one(
            feature_log_sample_rate=0.0,
            out_name=f"online_latency_no_logging_trial{trial}.json",
            trial=trial,
        )
        for trial in range(1, trials + 1)
    ]

    def spread(label: str, stats: list[dict[str, float | int]], key: str) -> None:
        values = sorted(float(s[key]) for s in stats)
        print(
            f"{label:28} {key}: {values[0]:.2f} - {values[-1]:.2f} ms across {len(values)} trials"
        )

    print()
    for key in ("median", "p90", "p99"):
        spread("with logging", with_logging, key)
        spread("without logging", without_logging, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
