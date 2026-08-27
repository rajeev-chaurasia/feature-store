"""P5: does what was served match what the definition says it should have been?

**The comparison this module makes is deliberately not the same one P2 makes.** P2 asks
"does event-time backfilling leak", by comparing ``as_of_event_time`` against
``as_of_known``, two different semantics computed by the same Spark code. This module
asks a different question: "does the online path, which is supposed to implement
``as_of_event_time`` semantics via tiles and a Redis head, actually agree with the batch
path's implementation of that same semantics." The recomputation this detector checks
served values against is therefore ``offline.pit.training_features`` called with
``semantics=Semantics.EVENT_TIME`` (the tiled, event-time path, with the same freshness
gating the served values already went through), **not** ``semantics=Semantics.KNOWN``
(the strict, created-time-filtered reference). Comparing against the strict semantics
instead would conflate a real implementation bug with the leak P2 already measures and
reports separately, and every mismatch this detector found would be uninterpretable: a
reader could never tell whether a reported skew was a bug or just late data.

Because both sides of the primary comparison intend the *same* semantics, any
disagreement beyond float tolerance is either a genuine implementation divergence, or the
online path's own answer disagreeing with itself because of something the tiled semantics
did not intend (evicted state, a dropped write). This is what "the shared definition
eliminates the implementation class of skew by construction, so what remains is the
residual" means in code: the tiled recomputation and the online consumer are two
independent implementations of one shared spec (``asofline.agg``, ``compiler.spec``), and
this module is the check that they still agree.

**A second, finer comparison against the strict semantics** (``known_features``) is used
only to classify a mismatch, not to define one: if the tiled recompute and the strict
recompute already disagree with each other on a row, then some of that row's served-vs-
recompute gap is attributable to late-arriving data interacting with this specific row
(the same mechanism P2 measures, now observed at the level of one served request) rather
than to a defect in the online implementation. Both cases are still real, but a reader
needs to be able to tell them apart, and the plan calls this out explicitly: "a detector
that says '3.4% mismatch' is an alarm; a detector that says '3.4%, of which 3.1% is late
data and 0.3% is float order' is a tool."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from asofline.compiler.batch import Strategy
from asofline.compiler.spec import feature_specs
from asofline.definitions.view import FeatureView
from asofline.offline.pit import Semantics, training_features

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

FLOAT_TOLERANCE_REL = 1e-6
"""Relative tolerance below which a served/recomputed gap is float noise, not skew.

``compiler.batch``'s own docstring documents why PREFIX and RANGE do not agree bit for
bit: subtracting two running totals is not the same sequence of float operations as
folding the covered tiles. The online consumer's own read-modify-write merge order is a
third such sequence. All three are the same monoid law-abiding arithmetic taking a
different path to it, and this tolerance is generous enough to absorb that without also
absorbing a real bug: the injected-bug done-test's gap (a whole head window's contribution
missing) is orders of magnitude larger than accumulated rounding ever produces.
"""

BUCKET_MATCH = "match"
BUCKET_ONE_SIDE_NULL = "one_side_null"
BUCKET_PARTIAL_HEAD_TILE = "partial_head_tile"
BUCKET_LATE_ARRIVING_DATA = "late_arriving_data"


def _close_enough(a: float, b: float, *, rel_tol: float = FLOAT_TOLERANCE_REL) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def classify(served: float | None, tiled: float | None, known: float | None) -> str:
    """One row's bucket, given the served value and both recomputations.

    ``tiled`` is the primary ground truth (see the module docstring); ``known`` is
    consulted only to decide whether a real gap is attributable to the P2-style leak
    interacting with this particular row, versus being unexplained by semantics at all
    and therefore a genuine implementation defect.
    """
    if (served is None) != (tiled is None):
        return BUCKET_ONE_SIDE_NULL
    if served is None or tiled is None:
        return BUCKET_MATCH  # both null: nothing served, nothing expected, no disagreement
    if _close_enough(served, tiled):
        return BUCKET_MATCH
    if known is not None and _close_enough(tiled, known):
        # The tiled and strict recomputations agree with each other on this row, so
        # nothing about late-arriving data explains the gap: the served value itself is
        # simply wrong relative to what both offline computations say it should be.
        return BUCKET_PARTIAL_HEAD_TILE
    return BUCKET_LATE_ARRIVING_DATA


def load_served_wide(
    spark: SparkSession,
    view: FeatureView,
    log_table: str,
    *,
    min_request_ts_ms: int | None = None,
    max_request_ts_ms: int | None = None,
) -> DataFrame:
    """One row per served request, pivoted back from the long ``serving_log`` shape.

    ``log_id`` is what identifies one served request: every exploded row from one
    ``FeatureLogEntry`` carries the same ``log_id``, so pivoting on it (rather than on the
    entity/timestamp pair) needs no assumption about how many join keys the view has.

    The optional time window is not a testing convenience bolted on after the fact: a real
    detector run is naturally scoped to a period (a day's traffic, an hour since a
    deploy), and ``serving_log`` is one long-lived, ever-growing table shared by every
    request the serving layer has ever logged. Without a window, a detector run drifts
    further from "compare what was recently served" the longer the table lives, and every
    run gets slower for no benefit. Bounds are compared via ``unix_millis`` rather than a
    driver-side timestamp, for the same reason every other timestamp comparison in this
    project is: see ``tests/spark/test_smoke_iceberg.py`` for what a driver-side comparison
    silently gets wrong.
    """
    specs = feature_specs(view)
    join_keys = ", ".join(f"MAX(entity_keys['{key}']) AS {key}" for key in view.join_keys)
    pivots = ", ".join(
        f"MAX(CASE WHEN feature_name = '{spec.feature_name}' THEN value END) AS {spec.feature_name}"
        for spec in specs
    )
    window_predicates = []
    if min_request_ts_ms is not None:
        window_predicates.append(f"unix_millis(request_ts) >= {min_request_ts_ms}")
    if max_request_ts_ms is not None:
        window_predicates.append(f"unix_millis(request_ts) <= {max_request_ts_ms}")
    window_clause = "".join(f" AND {predicate}" for predicate in window_predicates)
    return spark.sql(
        f"""
        SELECT log_id, {join_keys},
               MAX(request_ts) AS as_of_ts,
               {pivots}
        FROM {log_table}
        WHERE view_name = '{view.name}' AND view_version = {view.version}{window_clause}
        GROUP BY log_id
        """
    )


def _entity_frame_for_backfill(served_wide: DataFrame, view: FeatureView) -> DataFrame:
    columns = [*view.join_keys, "as_of_ts"]
    return served_wide.select(*columns).dropDuplicates()


@dataclass(frozen=True, slots=True)
class FeatureSkewReport:
    """Everything reported about one feature's agreement with its own recomputation."""

    feature_name: str
    compared: int
    served_null_count: int
    recomputed_null_count: int
    bucket_counts: dict[str, int] = field(default_factory=dict)
    psi: float | None = None
    ks_statistic: float | None = None

    @property
    def mismatch_count(self) -> int:
        return sum(count for bucket, count in self.bucket_counts.items() if bucket != BUCKET_MATCH)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatch_count / self.compared if self.compared else 0.0

    @property
    def served_null_rate(self) -> float:
        return self.served_null_count / self.compared if self.compared else 0.0

    @property
    def recomputed_null_rate(self) -> float:
        return self.recomputed_null_count / self.compared if self.compared else 0.0

    @property
    def null_rate_delta(self) -> float:
        return self.served_null_rate - self.recomputed_null_rate

    def bucket_rate(self, bucket: str) -> float:
        return self.bucket_counts.get(bucket, 0) / self.compared if self.compared else 0.0


def population_stability_index(
    expected: list[float], actual: list[float], *, bins: int = 10
) -> float | None:
    """PSI between two numeric samples, binned on the pooled sample's own quantiles.

    Quantile bins rather than fixed-width ones, because a feature's scale is arbitrary
    (seconds, counts) and a fixed-width bin would put nearly everything in one bucket for
    a skewed distribution, making PSI meaningless for exactly the count/sum features this
    project mostly serves. ``None`` when there are too few distinct values to bin at all,
    rather than a fabricated number.
    """
    pooled = sorted(expected + actual)
    if len(pooled) < bins * 2:
        return None
    edges = sorted({pooled[int(q * (len(pooled) - 1))] for q in (i / bins for i in range(1, bins))})
    if not edges:
        return None

    def bucket_shares(values: list[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for value in values:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(values) or 1
        return [count / total for count in counts]

    expected_shares = bucket_shares(expected)
    actual_shares = bucket_shares(actual)
    psi = 0.0
    floor = 1e-6  # avoids log(0)/div-by-0 for a bin either sample never landed in
    for e, a in zip(expected_shares, actual_shares, strict=True):
        e, a = max(e, floor), max(a, floor)
        psi += (a - e) * math.log(a / e)
    return psi


def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float | None:
    """The two-sample Kolmogorov-Smirnov statistic: the largest gap between empirical CDFs.

    Implemented directly rather than pulled from scipy, since this project has no other
    dependency on it: the definition is short enough that adding a dependency for it would
    cost more than writing it.
    """
    if not sample_a or not sample_b:
        return None
    a_sorted, b_sorted = sorted(sample_a), sorted(sample_b)
    all_values = sorted(set(a_sorted) | set(b_sorted))
    n_a, n_b = len(a_sorted), len(b_sorted)
    max_gap = 0.0
    for value in all_values:
        cdf_a = sum(1 for x in a_sorted if x <= value) / n_a
        cdf_b = sum(1 for x in b_sorted if x <= value) / n_b
        max_gap = max(max_gap, abs(cdf_a - cdf_b))
    return max_gap


def detect_skew(
    spark: SparkSession,
    view: FeatureView,
    *,
    namespace: str,
    raw_table: str,
    log_table: str,
    min_request_ts_ms: int | None = None,
    max_request_ts_ms: int | None = None,
) -> dict[str, FeatureSkewReport]:
    """Compare every served request in ``log_table`` for ``view`` against a recomputation.

    Returns one :class:`FeatureSkewReport` per feature the view serves. See
    ``load_served_wide`` for why the time window matters: without it, this compares
    against every request the serving layer has ever logged, not the run this call is
    actually meant to audit.
    """
    served_wide = load_served_wide(
        spark,
        view,
        log_table,
        min_request_ts_ms=min_request_ts_ms,
        max_request_ts_ms=max_request_ts_ms,
    )
    entity_frame = _entity_frame_for_backfill(served_wide, view)

    # Both recomputations go through the same entry point with the same freshness gating
    # (apply_ttl=True, the default), so a served null caused by legitimate staleness is
    # expected to match a recomputed null caused by the same rule, and only a genuine
    # disagreement about a non-stale value's contents, or about whether the entity is
    # stale at all, shows up as a mismatch.
    tiled = training_features(
        spark,
        view,
        entity_frame,
        namespace=namespace,
        raw_table=raw_table,
        semantics=Semantics.EVENT_TIME,
        strategy=Strategy.PREFIX,
    )
    known = training_features(
        spark,
        view,
        entity_frame,
        namespace=namespace,
        raw_table=raw_table,
        semantics=Semantics.KNOWN,
        strategy=Strategy.PREFIX,
    )

    join_keys = view.join_keys
    served_wide.createOrReplaceTempView("_asofline_skew_served")
    tiled.createOrReplaceTempView("_asofline_skew_tiled")
    known.createOrReplaceTempView("_asofline_skew_known")

    join_on_tiled = " AND ".join(f"s.{key} = t.{key}" for key in join_keys)
    join_on_known = " AND ".join(f"s.{key} = k.{key}" for key in join_keys)
    specs = feature_specs(view)
    columns = ", ".join(
        f"s.{spec.feature_name} AS served_{spec.feature_name}, "
        f"t.{spec.feature_name} AS tiled_{spec.feature_name}, "
        f"k.{spec.feature_name} AS known_{spec.feature_name}"
        for spec in specs
    )
    joined = spark.sql(
        f"""
        SELECT s.log_id, {columns}
        FROM _asofline_skew_served s
        JOIN _asofline_skew_tiled t ON {join_on_tiled} AND s.as_of_ts = t.as_of_ts
        JOIN _asofline_skew_known k ON {join_on_known} AND s.as_of_ts = k.as_of_ts
        """
    ).collect()

    reports: dict[str, FeatureSkewReport] = {}
    for spec in specs:
        served_values = [row[f"served_{spec.feature_name}"] for row in joined]
        tiled_values = [row[f"tiled_{spec.feature_name}"] for row in joined]
        known_values = [row[f"known_{spec.feature_name}"] for row in joined]

        bucket_counts: dict[str, int] = {}
        for served, tiled_value, known_value in zip(
            served_values, tiled_values, known_values, strict=True
        ):
            bucket = classify(served, tiled_value, known_value)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        served_numeric = [v for v in served_values if v is not None]
        tiled_numeric = [v for v in tiled_values if v is not None]
        reports[spec.feature_name] = FeatureSkewReport(
            feature_name=spec.feature_name,
            compared=len(joined),
            served_null_count=sum(1 for v in served_values if v is None),
            recomputed_null_count=sum(1 for v in tiled_values if v is None),
            bucket_counts=bucket_counts,
            psi=population_stability_index(tiled_numeric, served_numeric),
            ks_statistic=ks_statistic(tiled_numeric, served_numeric),
        )
    return reports
