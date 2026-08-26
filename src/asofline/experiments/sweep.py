"""How the cost of event-time backfilling scales with how late data actually arrives.

A single disagreement rate from one arbitrary late-tail setting is a weak claim. The
first run of ``run_leakage_experiment`` produced 0.1%, which reads as "this does not
matter", and the reason is arithmetic rather than anything about feature stores:

    P(a given entity row disagrees) ~ events_per_entity * mean_in_flight_time / stream_span

At a 95-second median late delay over a ten-day stream, that is about 1e-3 no matter how
correct or incorrect the implementation is. The interesting question is not "what is the
number" but "where is the threshold", so this sweeps the late tail and reports the curve.

The sweep varies ``late_delay_scale_ms``, which sets how long a late event stays in
flight. Real systems sit at very different points on this axis: a server-side event bus
is seconds, a mobile client that buffers while offline is minutes to hours, and a
cross-region batch upload is hours. Reporting the curve lets a reader find their own
system on it instead of taking one lab number on faith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from asofline.definitions.view import FeatureView
from asofline.demo.generator import EngagementGenerator, GeneratorConfig
from asofline.experiments.leakage import LeakageResult, run_leakage_experiment
from asofline.offline.ingest import load_events
from asofline.offline.tiles import build_tiles

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@dataclass(frozen=True, slots=True)
class SweepPoint:
    late_delay_scale_ms: int
    observed_p50_lateness_ms: int
    observed_p99_lateness_ms: int
    observed_max_lateness_ms: int
    result: LeakageResult


def _lateness_quantiles(lateness: list[int]) -> tuple[int, int, int]:
    ordered = sorted(lateness)
    n = len(ordered)
    return ordered[n // 2], ordered[min(n - 1, int(0.99 * n))], ordered[-1]


def run_lateness_sweep(
    spark: SparkSession,
    view: FeatureView,
    *,
    namespace: str,
    base_config: GeneratorConfig,
    scales_ms: tuple[int, ...],
    rows: int = 20_000,
    seed: int = 20260823,
) -> list[SweepPoint]:
    """One full regeneration, ingest, tile build and experiment per point.

    Regenerating rather than perturbing arrival times in place: the generator is the only
    thing that knows how to produce a self-consistent stream, and a stream patched
    afterwards is one no ``GeneratorConfig`` describes.
    """
    from dataclasses import replace

    points: list[SweepPoint] = []
    for scale_ms in scales_ms:
        config = replace(base_config, late_delay_scale_ms=scale_ms)
        events = EngagementGenerator(config).generate()
        p50, p99, worst = _lateness_quantiles([event.lateness_ms for event in events])

        table = f"engagement_events_scale_{scale_ms}"
        raw = load_events(spark, events, namespace=namespace, table=table)
        build_tiles(spark, view, namespace=namespace, raw_table=raw)
        result = run_leakage_experiment(
            spark, view, namespace=namespace, raw_table=raw, rows=rows, seed=seed
        )
        points.append(
            SweepPoint(
                late_delay_scale_ms=scale_ms,
                observed_p50_lateness_ms=p50,
                observed_p99_lateness_ms=p99,
                observed_max_lateness_ms=worst,
                result=result,
            )
        )
    return points
