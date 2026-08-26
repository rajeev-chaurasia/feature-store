"""Run P4's freshness probe for real and commit the resulting evidence artifact.

    uv run python scripts/run_freshness_probe.py
    uv run python scripts/run_freshness_probe.py --iterations 200 --run-id my-run

Freshness here means **event-time to visible-in-serving-response** (see
``asofline.bench.freshness``), not consumer lag. Each sample is the wall-clock gap between
a probe event's own ``event_ts`` and the first online-store read that reflects its
contribution to ``watch_seconds_sum_1h`` on ``user_engagement``, with a real,
separately-running ``streaming.consumer`` instance doing the actual Kafka-to-Redis work in
between. The written artifact validates against ``asofline.artifacts`` and
``scripts/validate_artifacts.py`` exactly like a P3 online-latency run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from asofline.artifacts import ARTIFACT_KIND_STREAMING_LAG, build_artifact, write_artifact_atomic
from asofline.bench.freshness import (
    DEFAULT_FEATURE_NAME,
    FreshnessProbeConfig,
    run_freshness_probe_with_consumer,
)
from asofline.config import SETTINGS
from asofline.demo.views import USER_ENGAGEMENT

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--probe-timeout-s", type=float, default=5.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.03)
    parser.add_argument("--warmup-timeout-s", type=float, default=30.0)
    parser.add_argument("--run-id", default="p4-streaming-freshness")
    args = parser.parse_args(argv)

    config = FreshnessProbeConfig(
        iterations=args.iterations,
        kafka_bootstrap=SETTINGS.kafka_bootstrap,
        events_topic=SETTINGS.events_topic,
        redis_url=SETTINGS.redis_url,
        poll_interval_s=args.poll_interval_s,
        probe_timeout_s=args.probe_timeout_s,
        warmup_timeout_s=args.warmup_timeout_s,
    )

    print(
        f"running {config.iterations} freshness probes against "
        f"kafka={config.kafka_bootstrap} redis={config.redis_url} topic={config.events_topic}"
    )
    result = run_freshness_probe_with_consumer(
        config, view=USER_ENGAGEMENT, feature_name=DEFAULT_FEATURE_NAME
    )

    if not result.raw_samples_ms:
        print(
            f"every probe timed out ({result.failure_count}/{result.requested_count} failed); "
            "no evidence to commit",
            file=sys.stderr,
        )
        return 1

    artifact = build_artifact(
        artifact_kind=ARTIFACT_KIND_STREAMING_LAG,
        created_at=datetime.now(UTC).isoformat(),
        config={
            "iterations": config.iterations,
            "kafka_bootstrap": config.kafka_bootstrap,
            "events_topic": config.events_topic,
            "redis_url": config.redis_url,
            "poll_interval_s": config.poll_interval_s,
            "probe_timeout_s": config.probe_timeout_s,
            "warmup_timeout_s": config.warmup_timeout_s,
            "view": USER_ENGAGEMENT.name,
            "feature_name": DEFAULT_FEATURE_NAME,
            "watch_seconds_value": config.watch_seconds_value,
            "requested_count": result.requested_count,
            "failure_count": result.failure_count,
            "measurement": "event-time to visible-in-serving-response, not consumer lag",
        },
        raw_samples_ms=result.raw_samples_ms,
    )

    path = RESULTS_ROOT / args.run_id / "streaming_lag.json"
    write_artifact_atomic(artifact, path)

    stats = artifact["statistics"]
    print(f"wrote {path}")
    print(
        f"count={stats['count']} failures={result.failure_count}/{result.requested_count} "
        f"min={stats['min']:.1f}ms p50(median)={stats['median']:.1f}ms "
        f"p90={stats['p90']:.1f}ms p99={stats['p99']:.1f}ms max={stats['max']:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
