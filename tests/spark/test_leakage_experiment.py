"""P2's second done-test: measure the leak, don't just prove it exists.

The adversarial test in test_point_in_time.py proves KNOWN excludes a late arrival that
EVENT_TIME includes, on one hand-built example. This exercises the same mechanism at
population scale on the generator's late tail, and checks the measurement responds to the
knob that should move it, without pinning a specific AUC number as a golden value: fitting
a model on generated data is not something a unit test should assert to four decimal
places, and doing so would make the test fragile to the RNG rather than to the mechanism.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from pyspark.sql import SparkSession

from asofline.demo.generator import EngagementGenerator, GeneratorConfig
from asofline.demo.views import USER_ENGAGEMENT as VIEW
from asofline.experiments.leakage import run_leakage_experiment
from asofline.offline.ingest import load_events
from asofline.offline.tiles import build_tiles

pytestmark = pytest.mark.spark

NAMESPACE = "asofline_leakage_test"
BASE_CONFIG = GeneratorConfig(n_events=60_000, n_users=800, n_videos=3_000)


def _seed_stream(spark: SparkSession, config: GeneratorConfig, table: str) -> str:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    events = EngagementGenerator(config).generate()
    raw = load_events(spark, events, namespace=NAMESPACE, table=table)
    build_tiles(spark, VIEW, namespace=NAMESPACE, raw_table=raw)
    return raw


@pytest.fixture(scope="module")
def wide_tail(spark: SparkSession) -> Iterator[str]:
    config = replace(BASE_CONFIG, late_delay_scale_ms=20 * 60 * 1000)
    yield _seed_stream(spark, config, "wide_tail_events")


@pytest.fixture(scope="module")
def no_tail(spark: SparkSession) -> Iterator[str]:
    config = replace(BASE_CONFIG, late_fraction=0.0)
    yield _seed_stream(spark, config, "no_tail_events")


def test_a_wide_late_tail_produces_measurable_disagreement(
    spark: SparkSession, wide_tail: str
) -> None:
    result = run_leakage_experiment(
        spark, VIEW, namespace=NAMESPACE, raw_table=wide_tail, rows=5_000
    )
    assert result.entity_rows == 5_000
    assert result.disagreement_rate > 0.02, (
        f"expected a wide late tail to produce noticeable disagreement, "
        f"got {result.disagreement_rate:.4f}"
    )
    assert result.features_compared == 8
    assert set(result.scores) == {"leaky_offline", "leaky_in_production", "strict"}
    for score in result.scores.values():
        assert 0.0 <= score.auc <= 1.0


def test_no_late_tail_means_no_disagreement(spark: SparkSession, no_tail: str) -> None:
    """The false-positive check for this measurement.

    With late_fraction=0, EVENT_TIME and KNOWN read the identical set of events for every
    window, so the two modes must agree exactly. If they did not, the measurement itself
    would be biased and every number in the docstring would be suspect.
    """
    result = run_leakage_experiment(spark, VIEW, namespace=NAMESPACE, raw_table=no_tail, rows=3_000)
    assert result.disagreement_rate == 0.0
    assert result.offline_to_production_gap == pytest.approx(0.0, abs=1e-9)


def test_disagreement_rate_increases_with_the_late_tail_width(spark: SparkSession) -> None:
    """The monotonic curve the headline number rests on, at two points rather than five.

    The full five-point sweep in asofline.experiments.sweep takes several minutes and
    belongs in the committed results, not in a test that runs on every push. This checks
    the direction of the effect cheaply: narrow tail, then wide tail, same seed, same
    entity sample size, and the wide one must disagree at least as often.
    """
    narrow_raw = _seed_stream(
        spark, replace(BASE_CONFIG, late_delay_scale_ms=1_000), "narrow_tail_events"
    )
    wide_raw = _seed_stream(
        spark, replace(BASE_CONFIG, late_delay_scale_ms=30 * 60 * 1000), "very_wide_tail_events"
    )
    narrow = run_leakage_experiment(
        spark, VIEW, namespace=NAMESPACE, raw_table=narrow_raw, rows=4_000, seed=99
    )
    wide = run_leakage_experiment(
        spark, VIEW, namespace=NAMESPACE, raw_table=wide_raw, rows=4_000, seed=99
    )
    assert wide.disagreement_rate > narrow.disagreement_rate
