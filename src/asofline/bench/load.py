"""An open-loop load generator against the online serving API.

**Open-loop, not closed-loop.** A closed-loop generator waits for one response before
sending the next, which means the arrival rate it produces is capped by whatever the
server's own latency is: a slow server automatically throttles the offered load, and the
generator can never actually observe what happens under a fixed rate of demand. Requests
here are scheduled at a fixed rate regardless of whether earlier ones have completed.

**Latency is measured from the intended send time, not the actual one.** Each request has
a scheduled instant, ``start + i / qps``. If the event loop is busy and a request goes out
late, the delay between intended and actual dispatch is queueing delay, and it belongs in
the measurement: a real client issuing requests on a schedule experiences exactly that
delay, and discarding it is the classic way a benchmark flatters itself (coordinated
omission). Latency for request ``i`` is therefore
``(response_received_at - intended_send_at)``, not ``(response_received_at -
actually_sent_at)``.

Every completed request's latency is kept. This project does not summarize before it has
to: ``asofline.artifacts.build_artifact`` is what turns ``raw_samples_ms`` into
statistics, and it does that from the full list, not from anything this module precomputes.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True, slots=True)
class LoadTestConfig:
    """Everything that determines the offered load, in one hashable object.

    Committed alongside the results (as ``config`` in the artifact), so a published
    latency number names the exact load that produced it.
    """

    base_url: str
    view: str
    qps: float
    duration_s: float
    entity_pool_size: int
    seed: int = 20260823
    request_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.qps <= 0:
            raise ValueError(f"qps must be positive, got {self.qps}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.entity_pool_size < 1:
            raise ValueError(f"entity_pool_size must be positive, got {self.entity_pool_size}")

    @property
    def request_count(self) -> int:
        return round(self.qps * self.duration_s)


@dataclass(frozen=True, slots=True)
class LoadTestResult:
    raw_samples_ms: list[float]
    error_count: int
    requested_count: int

    @property
    def completed_count(self) -> int:
        return len(self.raw_samples_ms)


def intended_send_times(config: LoadTestConfig) -> list[float]:
    """The schedule: request ``i`` is due at ``i / qps`` seconds after start.

    A pure function of the config, with no clock read inside it, so the schedule itself is
    exactly reproducible and testable without an event loop.
    """
    return [i / config.qps for i in range(config.request_count)]


def sample_entity_pool(config: LoadTestConfig, join_key: str) -> list[dict[str, str]]:
    """A fixed, seeded pool of entity requests to draw from during the run.

    Reused across the whole run rather than generated fresh per request: real traffic
    revisits the same hot entities repeatedly, which is also what the online store's
    Redis footprint and cache behaviour actually see, and a pool drawn once with a fixed
    seed keeps the exact set of entities probed part of the reproducible configuration.
    """
    rng = random.Random(config.seed)
    return [{join_key: f"u{rng.randrange(10_000_000):07d}"} for _ in range(config.entity_pool_size)]


@dataclass(slots=True)
class _Sink:
    """Collects results as requests complete, in arbitrary completion order.

    Order does not matter here: each sample is a self-contained latency measurement, and
    ``asofline.artifacts.compute_statistics`` does not care what order it receives them in.
    """

    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0


async def _fire(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    entities: list[dict[str, str]],
    intended_at: float,
    monotonic_start: float,
    sink: _Sink,
) -> None:
    payload: dict[str, Any] = {"view": config.view, "entities": entities}
    try:
        response = await client.post(
            "/get-online-features", json=payload, timeout=config.request_timeout_s
        )
        response.raise_for_status()
        response.json()  # Force the body to be read, so latency includes deserialisation.
    except Exception:
        sink.errors += 1
        return
    completed_at = time.perf_counter() - monotonic_start
    sink.latencies_ms.append((completed_at - intended_at) * 1000.0)


async def run_load_test(
    client: httpx.AsyncClient, config: LoadTestConfig, join_key: str
) -> LoadTestResult:
    """Run the schedule against ``client`` and return every completed latency sample.

    ``client`` is injected rather than constructed here so a test can point it at a
    ``httpx.MockTransport`` instead of a live server, exercising the scheduling and
    statistics-collection logic with no network and no running serving process at all.
    """
    entities = sample_entity_pool(config, join_key)
    rng = random.Random(config.seed + 1)
    schedule = intended_send_times(config)
    sink = _Sink()
    monotonic_start = time.perf_counter()

    tasks: list[asyncio.Task[None]] = []
    for intended_at in schedule:
        now = time.perf_counter() - monotonic_start
        if intended_at > now:
            await asyncio.sleep(intended_at - now)
        one_entity = [entities[rng.randrange(len(entities))]]
        tasks.append(
            asyncio.ensure_future(
                _fire(client, config, one_entity, intended_at, monotonic_start, sink)
            )
        )
    await asyncio.gather(*tasks)

    return LoadTestResult(
        raw_samples_ms=sink.latencies_ms,
        error_count=sink.errors,
        requested_count=config.request_count,
    )
