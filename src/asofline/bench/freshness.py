"""The freshness half of P4: event-time to visible-in-serving-response, measured for real.

Consumer lag (offset behind latest) is not this number. Lag is easy and flattering: it can
read zero while a consumer is mid-way through a batch's Redis writes, or once those writes
have landed but before anyone has actually read them back. What this module measures
instead is the plan's harder, more honest metric: publish one uniquely tagged event onto
Kafka, and time how long it takes before a read of the online store reflects that event's
own contribution. That number folds in Kafka's fetch batching, the consumer's poll loop,
the Redis read-modify-write, and this probe's own poll interval for noticing -- which is
exactly why it is slower than lag, and why it is the number worth publishing.

**Why a real, separately-running consumer, not a call into ``process_event``.** Calling
``streaming.consumer.process_event`` directly would time only the write-then-read hop and
skip the Kafka-consume hop entirely, which is where most of the real latency lives (poll
intervals, batch draining, rebalance). ``run_freshness_probe_with_consumer`` instead runs
``asofline.streaming.consumer.run`` in its own OS process (``_freshness_consumer_worker``)
for the duration of the probe run, so the measured number is the pipeline a real
deployment would run, not a shortcut through it.

**Subprocess, not a thread.** A background thread would still poll Kafka and write Redis
faithfully, but it would also share the GIL and this harness's own event loop with the
code trying to time it -- exactly the measurement interference ``bench/load.py`` already
goes out of its way to avoid on the client side. A subprocess is what a real deployment
actually looks like: two independent processes talking through Kafka and Redis, nothing
shared.

**Warmup, not a fixed sleep.** A freshly subscribed consumer group has to complete a Kafka
rebalance before its assignment is live, and the shared ``engagement_events`` topic can
carry a small backlog from earlier, unrelated runs; both are one-time startup costs, not
steady-state pipeline latency. Rather than guess a sleep duration and hope it is enough,
``run_freshness_probes`` publishes one throwaway warmup event first and blocks on it
becoming visible, with a generous timeout, before it starts timing anything that counts.
"""

from __future__ import annotations

import asyncio
import json
import math
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from confluent_kafka import KafkaError, Producer

from asofline.definitions.view import FeatureView
from asofline.demo.events import EngagementEvent, EventType
from asofline.demo.views import USER_ENGAGEMENT
from asofline.online.store import OnlineStore

DEFAULT_FEATURE_NAME = "watch_seconds_sum_1h"
DEFAULT_WATCH_SECONDS = 123.456

_WORKER_MODULE = "asofline.bench._freshness_consumer_worker"


@dataclass(frozen=True, slots=True)
class FreshnessProbeConfig:
    """Everything that determines one freshness-probe run, in one hashable object.

    Committed alongside the results (as ``config`` in the artifact), same discipline as
    ``bench.load.LoadTestConfig``: a published latency number must name the exact
    measurement that produced it.
    """

    iterations: int
    kafka_bootstrap: str
    events_topic: str
    redis_url: str
    poll_interval_s: float = 0.03
    probe_timeout_s: float = 5.0
    warmup_timeout_s: float = 30.0
    watch_seconds_value: float = DEFAULT_WATCH_SECONDS
    consumer_group_prefix: str = "asofline-freshness-probe"
    consumer_shutdown_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {self.poll_interval_s}")
        if self.probe_timeout_s <= 0:
            raise ValueError(f"probe_timeout_s must be positive, got {self.probe_timeout_s}")
        if self.warmup_timeout_s <= 0:
            raise ValueError(f"warmup_timeout_s must be positive, got {self.warmup_timeout_s}")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Every completed latency sample, plus how many probes never arrived.

    Mirrors ``bench.load.LoadTestResult``'s split between completed samples and failures:
    a probe that times out is recorded as a failure, never as a fabricated latency value.
    """

    raw_samples_ms: list[float]
    failure_count: int
    requested_count: int

    @property
    def completed_count(self) -> int:
        return len(self.raw_samples_ms)


def _build_probe_event(user_id: str, watch_seconds: float, event_ts_ms: int) -> EngagementEvent:
    """A watch event with a known, checkable contribution: exactly ``watch_seconds``.

    A fresh ``user_id`` per probe (never reused) means this event's contribution to
    ``watch_seconds_sum_1h`` can never be folded together with another probe's, or with
    anything else on the topic: the feature value for this entity is either ``None``
    (not yet visible) or exactly ``watch_seconds`` (visible), nothing in between.
    """
    return EngagementEvent(
        event_id=uuid.uuid4().hex,
        event_type=EventType.WATCH,
        user_id=user_id,
        video_id=f"v-{user_id}",
        event_ts=event_ts_ms,
        created_ts=event_ts_ms,
        watch_seconds=watch_seconds,
    )


def _publish(
    producer: Producer, topic: str, event: EngagementEvent, *, flush_timeout_s: float
) -> None:
    """Publish one event and confirm delivery before returning.

    ``EngagementEvent.to_dict()`` as JSON is the exact wire format both
    ``streaming.consumer`` and ``streaming.to_iceberg`` already expect. Blocking on
    ``flush`` (rather than firing and forgetting) means the delay between "produce called"
    and "actually on the broker" is real producer-side latency and is correctly folded
    into this probe's own measurement, not hidden from it.
    """
    delivery_errors: list[KafkaError] = []

    def _on_delivery(error: KafkaError | None, _message: object) -> None:
        if error is not None:
            delivery_errors.append(error)

    payload = json.dumps(event.to_dict()).encode("utf-8")
    producer.produce(topic, value=payload, callback=_on_delivery)
    still_queued = producer.flush(flush_timeout_s)
    if still_queued:
        raise RuntimeError(
            f"kafka producer flush timed out with {still_queued} message(s) still queued"
        )
    if delivery_errors:
        raise RuntimeError(
            f"kafka delivery failed for event {event.event_id}: {delivery_errors[0]}"
        )


async def _poll_until_visible(
    store: OnlineStore,
    view: FeatureView,
    entity: Mapping[str, str],
    feature_name: str,
    expected_value: float,
    *,
    timeout_s: float,
    poll_interval_s: float,
    tolerance_abs: float = 1e-6,
) -> float | None:
    """Poll the online store until ``feature_name`` reflects ``expected_value``.

    Returns the wall-clock epoch-millisecond instant the expected value was first
    observed, or ``None`` if ``timeout_s`` elapses first. Wall clock (``time.time()``),
    not ``time.monotonic()``, because the latency this module reports is measured against
    ``EngagementEvent.event_ts``, itself a wall-clock epoch-millisecond stamp set by the
    publisher; mixing a monotonic clock into one side of that subtraction would make the
    difference meaningless.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        [vector] = await store.get_online_features(view, [entity])
        value = vector.get(feature_name)
        if value is not None and math.isclose(
            value, expected_value, rel_tol=0.0, abs_tol=tolerance_abs
        ):
            return time.time() * 1000.0
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(poll_interval_s)


async def run_freshness_probes(
    config: FreshnessProbeConfig,
    *,
    view: FeatureView = USER_ENGAGEMENT,
    feature_name: str = DEFAULT_FEATURE_NAME,
) -> ProbeResult:
    """Run ``config.iterations`` probes against an already-running consumer.

    Does not start or stop the consumer itself; see ``run_freshness_probe_with_consumer``
    for the version that owns that lifecycle. Kept separate so a caller that already has a
    consumer running (this module's own integration test, timing more than one batch of
    probes against one subprocess) is not forced to pay subprocess startup twice.
    """
    producer = Producer({"bootstrap.servers": config.kafka_bootstrap})
    store = OnlineStore.from_url(config.redis_url)
    try:
        run_tag = uuid.uuid4().hex[:10]

        warmup_user = f"fresh-warmup-{run_tag}"
        warmup_event = _build_probe_event(
            warmup_user, config.watch_seconds_value, int(time.time() * 1000)
        )
        _publish(
            producer, config.events_topic, warmup_event, flush_timeout_s=config.warmup_timeout_s
        )
        warmup_seen_at = await _poll_until_visible(
            store,
            view,
            {"user_id": warmup_user},
            feature_name,
            config.watch_seconds_value,
            timeout_s=config.warmup_timeout_s,
            poll_interval_s=config.poll_interval_s,
        )
        if warmup_seen_at is None:
            raise RuntimeError(
                f"warmup probe never became visible within {config.warmup_timeout_s}s; "
                "the consumer is not running, has not joined its group, or cannot reach "
                "Redis/Kafka"
            )

        samples: list[float] = []
        failures = 0
        for i in range(config.iterations):
            user_id = f"fresh-{run_tag}-{i:06d}"
            event = _build_probe_event(user_id, config.watch_seconds_value, int(time.time() * 1000))
            _publish(producer, config.events_topic, event, flush_timeout_s=config.probe_timeout_s)
            observed_at_ms = await _poll_until_visible(
                store,
                view,
                {"user_id": user_id},
                feature_name,
                config.watch_seconds_value,
                timeout_s=config.probe_timeout_s,
                poll_interval_s=config.poll_interval_s,
            )
            if observed_at_ms is None:
                failures += 1
                continue
            samples.append(observed_at_ms - event.event_ts)

        return ProbeResult(
            raw_samples_ms=samples, failure_count=failures, requested_count=config.iterations
        )
    finally:
        producer.flush(5.0)
        await store.close()


def start_consumer_subprocess(
    *, kafka_bootstrap: str, events_topic: str, redis_url: str, group_id: str
) -> subprocess.Popen[bytes]:
    """Launch the Kafka-to-Redis consumer as its own OS process.

    Every endpoint is passed explicitly on the command line rather than inherited from
    ``SETTINGS``, so a probe can point the worker at a disposable Redis database and a
    run-scoped consumer group without touching ``asofline.streaming.consumer`` or
    colliding with a production instance using the real ``asofline-to-redis`` group.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            _WORKER_MODULE,
            "--kafka-bootstrap",
            kafka_bootstrap,
            "--topic",
            events_topic,
            "--redis-url",
            redis_url,
            "--group-id",
            group_id,
        ]
    )


def stop_consumer_subprocess(proc: subprocess.Popen[bytes], *, timeout_s: float = 5.0) -> None:
    """Shut the consumer subprocess down, and guarantee it is actually gone.

    SIGINT first: Python's default handler turns it into a ``KeyboardInterrupt`` in the
    worker's main thread, which unwinds through ``streaming.consumer.run``'s ``finally``
    clause and calls ``Consumer.close()``, leaving the group cleanly instead of waiting out
    a session timeout. If that has not worked within ``timeout_s`` (a hang, or a signal
    swallowed for some reason), ``kill`` guarantees no orphaned process survives this
    function regardless of what went wrong.
    """
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def run_freshness_probe_with_consumer(
    config: FreshnessProbeConfig,
    *,
    view: FeatureView = USER_ENGAGEMENT,
    feature_name: str = DEFAULT_FEATURE_NAME,
) -> ProbeResult:
    """Start a dedicated consumer subprocess, run the probe against it, then tear it down.

    Owns the subprocess's whole lifecycle in one ``try``/``finally`` so neither
    ``scripts/run_freshness_probe.py`` nor a test can forget to clean it up.
    """
    group_id = f"{config.consumer_group_prefix}-{uuid.uuid4().hex[:10]}"
    proc = start_consumer_subprocess(
        kafka_bootstrap=config.kafka_bootstrap,
        events_topic=config.events_topic,
        redis_url=config.redis_url,
        group_id=group_id,
    )
    try:
        return asyncio.run(run_freshness_probes(config, view=view, feature_name=feature_name))
    finally:
        stop_consumer_subprocess(proc, timeout_s=config.consumer_shutdown_timeout_s)
