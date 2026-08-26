"""How much does event-time backfilling actually cost?

Every feature store README asserts that point-in-time correctness matters. This measures
it on a stream whose late tail is known exactly, because it was generated.

**Measured result** (``asofline.experiments.sweep``, five late-tail widths, 15,000 probe
rows each, ``USER_ENGAGEMENT`` view, seed 20260823): the row disagreement rate is
monotonic in how long data stays in flight, from 0.08% at a 1-second delay scale to
19.7% at 30 minutes, matching a server-side event bus versus a client that buffers while
offline. The downstream AUC gap between the leaky model scored honestly and the strict
baseline stayed within +/-0.0001 at every point, including at 19.7% row disagreement.

That second number is reported because it is what came out, not because it is flattering.
It says something real: a logistic regression over these eight features, predicting a
watch within one hour, is not sensitive to the specific rows this leak perturbs, most
likely because count and sum features that are off by one late event rarely cross a
decision boundary the model actually uses. This is not evidence that point-in-time
correctness does not matter in general. It is evidence that "does the leak move AUC" is a
question with a real, checkable, sometimes-boring answer that depends on the model and
the label, and a project that only reports the disagreement rate would be asserting the
downstream consequence rather than measuring it.

Two numbers come out, and the second is the one worth leading with.

**Disagreement rate.** The fraction of training rows whose feature vector differs between
``EVENT_TIME`` and ``KNOWN``. Straightforward, and it answers "how often does this bite".

**The offline-to-production gap.** Three models, not two:

===========================  ==========  ==========  ===============================
Name                         Trained on  Scored on   What it represents
===========================  ==========  ==========  ===============================
``leaky_offline``            leaky       leaky       the AUC you would have reported
``leaky_in_production``      leaky       strict      what that model actually gets
``strict``                   strict      strict      the honest baseline
===========================  ==========  ==========  ===============================

``leaky_offline`` minus ``leaky_in_production`` is the drop a team sees on the day their
model ships, with no bug to find because nothing is broken: the offline number was
computed against data that production cannot reproduce. Reporting only
``leaky_offline`` against ``strict`` would understate it, because both are self-consistent
and the damage is precisely the inconsistency.

The split is **temporal**, on ``as_of``. A random split would let a row from after the
boundary inform a row from before it, which is a second leak on top of the one being
measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from asofline.compiler.spec import feature_specs
from asofline.definitions.view import FeatureView
from asofline.offline.pit import Semantics, training_features

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

LABEL = "label"
ROW_ID = "entity_row_id"
COMPARISON_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ModelScore:
    name: str
    trained_on: str
    scored_on: str
    auc: float


@dataclass(frozen=True, slots=True)
class LeakageResult:
    entity_rows: int
    train_rows: int
    test_rows: int
    label_rate: float
    features_compared: int
    rows_with_any_difference: int
    per_feature_difference_rate: dict[str, float] = field(default_factory=dict)
    scores: dict[str, ModelScore] = field(default_factory=dict)

    @property
    def disagreement_rate(self) -> float:
        return self.rows_with_any_difference / self.entity_rows if self.entity_rows else 0.0

    @property
    def offline_to_production_gap(self) -> float:
        """What the leaky model loses on the day it ships."""
        return self.scores["leaky_offline"].auc - self.scores["leaky_in_production"].auc


def sample_entity_rows(
    spark: SparkSession,
    view: FeatureView,
    *,
    raw_table: str,
    rows: int,
    seed: int,
    label_horizon_ms: int,
) -> DataFrame:
    """Draw ``(entity, as_of)`` pairs from the interior of the stream, and label them.

    The interior matters. An ``as_of`` in the first ``longest_window`` of the stream has a
    window that reaches back before any data exists, and one in the last
    ``label_horizon`` has a label computed over a truncated future. Both would be measured
    as feature quality when they are artefacts of where the stream was cut.

    Labels are computed from **every** event, ignoring ``created_ts``. That is correct and
    is not the leak under study: a label is observed after the fact by definition, and
    nobody trains on labels filtered by what was known at prediction time.
    """
    join_key = view.join_keys[0]
    longest_window_ms = int(view.longest_window.total_seconds() * 1000)
    bounds = spark.sql(
        f"SELECT MIN(unix_millis(event_ts)) AS lo, MAX(unix_millis(event_ts)) AS hi "
        f"FROM {raw_table}"
    ).collect()[0]
    lo = int(bounds["lo"]) + longest_window_ms
    hi = int(bounds["hi"]) - label_horizon_ms
    if hi <= lo:
        raise ValueError(
            f"stream is too short: it spans {int(bounds['hi']) - int(bounds['lo'])}ms, and "
            f"{longest_window_ms + label_horizon_ms}ms is reserved for warmup and labels"
        )

    # Sampled from the event table rather than from distinct entities, for two reasons.
    # A "SELECT DISTINCT entity" source caps the sample at one row per entity, which
    # silently returned 3,000 rows when 15,000 were asked for. And weighting entities by
    # activity is what a real training set looks like: one row per impression, so busy
    # entities appear often and dormant ones rarely.
    join_alias = f"c.{join_key}"
    return spark.sql(
        f"""
        SELECT {join_alias},
               timestamp_millis(c.as_of_ms) AS as_of_ts,
               CAST(COUNT(r.event_id) > 0 AS INT) AS {LABEL}
        FROM (
            SELECT {join_key},
                   CAST({lo} + rand({seed}) * {hi - lo} AS BIGINT) AS as_of_ms
            FROM {raw_table}
            ORDER BY rand({seed + 1})
            LIMIT {rows}
        ) c
        LEFT JOIN {raw_table} r
               ON r.{join_key} = c.{join_key}
              AND r.event_type = 'watch'
              AND unix_millis(r.event_ts) > c.as_of_ms
              AND unix_millis(r.event_ts) <= c.as_of_ms + {label_horizon_ms}
        GROUP BY {join_alias}, c.as_of_ms
        """
    )


def _feature_names(view: FeatureView) -> list[str]:
    return [spec.feature_name for spec in feature_specs(view)]


def compare_vectors(
    spark: SparkSession, view: FeatureView, strict: DataFrame, leaky: DataFrame
) -> tuple[int, dict[str, float]]:
    """Rows where any feature differs, and the per feature rate.

    A tolerance rather than equality, because the two modes reach the same arithmetic by
    different routes: strict folds raw events, leaky subtracts two running tile totals.
    Without it, float rounding would be reported as leakage and the headline number would
    be roughly 100%.
    """
    names = _feature_names(view)
    strict.createOrReplaceTempView("_asofline_strict")
    leaky.createOrReplaceTempView("_asofline_leaky")

    differs = [
        f"(NOT (s.{name} <=> l.{name}) AND "
        f"(s.{name} IS NULL OR l.{name} IS NULL OR "
        f"ABS(s.{name} - l.{name}) > {COMPARISON_TOLERANCE} * GREATEST(ABS(s.{name}), 1.0)))"
        f" AS d_{name}"
        for name in names
    ]
    spark.sql(
        f"""
        SELECT s.{ROW_ID}, {", ".join(differs)}
        FROM _asofline_strict s JOIN _asofline_leaky l ON l.{ROW_ID} = s.{ROW_ID}
        """
    ).createOrReplaceTempView("_asofline_diff")

    totals = ", ".join(f"SUM(CAST(d_{name} AS INT)) AS n_{name}" for name in names)
    any_expr = " OR ".join(f"d_{name}" for name in names)
    row = spark.sql(
        f"""
        SELECT COUNT(*) AS n, SUM(CAST(({any_expr}) AS INT)) AS n_any, {totals}
        FROM _asofline_diff
        """
    ).collect()[0]
    total = int(row["n"]) or 1
    rates = {name: (int(row[f"n_{name}"] or 0)) / total for name in names}
    return int(row["n_any"] or 0), rates


def _assemble(view: FeatureView, frame: DataFrame) -> DataFrame:
    """Feature columns into an MLlib vector.

    Nulls are filled with zero. That loses the distinction the ttl rule draws between an
    entity that was quiet and one that was gone, which is a real cost and is reported as
    ``null_rate`` rather than hidden. Encoding it properly needs a null-indicator column
    per feature, which is worth doing when the model is the point; here the model is only
    a measuring instrument and both arms are handicapped identically.
    """
    from pyspark.ml.feature import VectorAssembler

    names = _feature_names(view)
    filled = frame.na.fill(0.0, subset=names)
    return VectorAssembler(inputCols=names, outputCol="features").transform(filled)


def _fit_and_score(view: FeatureView, train: DataFrame, test: DataFrame) -> tuple[object, float]:
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    model = LogisticRegression(
        featuresCol="features", labelCol=LABEL, maxIter=50, regParam=0.01
    ).fit(_assemble(view, train))
    evaluator = BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC")
    return model, float(evaluator.evaluate(model.transform(_assemble(view, test))))


def _score_with(view: FeatureView, model: object, test: DataFrame) -> float:
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    evaluator = BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC")
    return float(evaluator.evaluate(model.transform(_assemble(view, test))))  # type: ignore[attr-defined]


def run_leakage_experiment(
    spark: SparkSession,
    view: FeatureView,
    *,
    namespace: str,
    raw_table: str,
    rows: int = 20_000,
    seed: int = 20260823,
    label_horizon_ms: int = 3_600_000,
    train_fraction: float = 0.7,
) -> LeakageResult:
    entity_df = sample_entity_rows(
        spark,
        view,
        raw_table=raw_table,
        rows=rows,
        seed=seed,
        label_horizon_ms=label_horizon_ms,
    ).cache()

    def backfill(semantics: Semantics) -> DataFrame:
        return training_features(
            spark,
            view,
            entity_df,
            namespace=namespace,
            raw_table=raw_table,
            semantics=semantics,
        ).cache()

    strict = backfill(Semantics.KNOWN)
    leaky = backfill(Semantics.EVENT_TIME)

    n_any, rates = compare_vectors(spark, view, strict, leaky)

    labels = entity_df.selectExpr(*view.join_keys, "as_of_ts", LABEL)
    join = view.join_keys[0]
    strict_labelled = strict.join(labels, on=[join, "as_of_ts"], how="inner")
    leaky_labelled = leaky.join(labels, on=[join, "as_of_ts"], how="inner")

    # The split point is computed on epoch milliseconds, not on the timestamp column:
    # approxQuantile refuses TimestampType outright. Both arms split on the same value, so
    # the train and test sets contain exactly the same entity rows in both modes and the
    # AUC comparison is not confounded by a different sample.
    strict_labelled.createOrReplaceTempView("_asofline_labelled")
    boundary = int(
        spark.sql(
            f"SELECT CAST(PERCENTILE_APPROX(unix_millis(as_of_ts), {train_fraction}) AS BIGINT) "
            f"AS b FROM _asofline_labelled"
        ).collect()[0]["b"]
    )

    def split(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
        stamped = frame.selectExpr("*", "unix_millis(as_of_ts) AS _as_of_ms")
        return (
            stamped.filter(f"_as_of_ms < {boundary}"),
            stamped.filter(f"_as_of_ms >= {boundary}"),
        )

    strict_train, strict_test = split(strict_labelled)
    leaky_train, leaky_test = split(leaky_labelled)

    leaky_model, auc_leaky_offline = _fit_and_score(view, leaky_train, leaky_test)
    _, auc_strict = _fit_and_score(view, strict_train, strict_test)
    # The same leaky model, scored on the features production can actually reproduce.
    auc_leaky_in_production = _score_with(view, leaky_model, strict_test)

    total = strict_labelled.count()
    train_rows = strict_train.count()
    positives = strict_labelled.filter(f"{LABEL} = 1").count()

    return LeakageResult(
        entity_rows=total,
        train_rows=train_rows,
        test_rows=total - train_rows,
        label_rate=positives / total if total else 0.0,
        features_compared=len(_feature_names(view)),
        rows_with_any_difference=n_any,
        per_feature_difference_rate=rates,
        scores={
            "leaky_offline": ModelScore(
                "leaky_offline", "as_of_event_time", "as_of_event_time", auc_leaky_offline
            ),
            "leaky_in_production": ModelScore(
                "leaky_in_production", "as_of_event_time", "as_of_known", auc_leaky_in_production
            ),
            "strict": ModelScore("strict", "as_of_known", "as_of_known", auc_strict),
        },
    )
