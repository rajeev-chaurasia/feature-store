"""The evidence-artifact schema, the validator, and proof the validator cannot be fooled.

This module has no Spark, Redis or Kafka dependency, so every test here runs on nothing
but the raw JSON schema, statistics and file I/O in ``asofline.artifacts``. The tamper
tests are the point: each one independently proves the validator catches one specific
kind of doctored file, because a validator that only passes on the happy path is not a
validator, it is a formatter.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from asofline.artifacts import (
    ARTIFACT_KIND_ONLINE_LATENCY,
    ARTIFACT_KIND_STREAMING_LAG,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    artifact_validation_errors,
    build_artifact,
    capture_environment,
    compute_statistics,
    write_artifact_atomic,
)


def _lognormal_samples(seed: int, n: int = 500) -> list[float]:
    rng = random.Random(seed)
    return [rng.lognormvariate(0.0, 0.5) * 10.0 for _ in range(n)]


def _valid_artifact() -> dict[str, Any]:
    return build_artifact(
        artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
        created_at="2026-08-25T12:00:00Z",
        config={"qps": 500, "duration_s": 60, "seed": 42},
        raw_samples_ms=_lognormal_samples(seed=42),
        environment={"python_version": "3.12.0", "platform": "test", "hostname": "ci"},
    )


class TestComputeStatisticsCorrectness:
    def test_hand_verifiable_percentiles_on_one_to_ten(self) -> None:
        stats = compute_statistics([float(v) for v in range(1, 11)])
        assert stats == {
            "count": 10,
            "min": 1.0,
            "max": 10.0,
            "mean": 5.5,
            "median": 5.5,
            "p90": 9.1,
            "p99": 9.91,
            "mad": 2.5,
        }

    def test_hand_verifiable_percentiles_on_odd_length_sample(self) -> None:
        stats = compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats == {
            "count": 5,
            "min": 1.0,
            "max": 5.0,
            "mean": 3.0,
            "median": 3.0,
            "p90": 4.6,
            "p99": 4.96,
            "mad": 1.0,
        }

    def test_single_sample_is_every_statistic(self) -> None:
        stats = compute_statistics([7.5])
        assert stats == {
            "count": 1,
            "min": 7.5,
            "max": 7.5,
            "mean": 7.5,
            "median": 7.5,
            "p90": 7.5,
            "p99": 7.5,
            "mad": 0.0,
        }

    def test_realistic_latency_data_is_internally_consistent(self) -> None:
        stats = compute_statistics(_lognormal_samples(seed=7))
        assert stats["min"] <= stats["median"] <= stats["p90"] <= stats["p99"] <= stats["max"]
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["count"] == 500
        assert stats["mad"] >= 0.0

    def test_rejects_empty_sample_list(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_statistics([])

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_samples(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            compute_statistics([1.0, 2.0, bad_value])


class TestBuildArtifact:
    def test_shape_and_derived_fields(self) -> None:
        artifact = _valid_artifact()
        assert artifact["schema_version"] == SCHEMA_VERSION
        assert artifact["artifact_kind"] == ARTIFACT_KIND_ONLINE_LATENCY
        assert artifact["created_at"] == "2026-08-25T12:00:00Z"
        assert artifact["config"] == {"qps": 500, "duration_s": 60, "seed": 42}
        assert artifact["statistics"] == compute_statistics(artifact["raw_samples_ms"])
        assert isinstance(artifact["raw_samples_sha256"], str)
        assert len(artifact["raw_samples_sha256"]) == 64

    def test_supports_streaming_lag_kind_via_the_same_schema(self) -> None:
        artifact = build_artifact(
            artifact_kind=ARTIFACT_KIND_STREAMING_LAG,
            created_at="2026-08-25T12:00:00Z",
            config={"topic": "engagement_events"},
            raw_samples_ms=[1.0, 2.0, 3.0],
        )
        assert artifact["artifact_kind"] == ARTIFACT_KIND_STREAMING_LAG
        assert not artifact_validation_errors(artifact)

    def test_never_reads_the_clock_created_at_is_verbatim(self) -> None:
        artifact = build_artifact(
            artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
            created_at="not-even-a-real-timestamp",
            config={},
            raw_samples_ms=[1.0],
        )
        assert artifact["created_at"] == "not-even-a-real-timestamp"

    def test_default_environment_has_the_documented_fields(self) -> None:
        environment = capture_environment()
        assert set(environment) == {
            "python_version",
            "platform",
            "hostname",
            "cpu_count",
            "in_docker",
        }
        assert "gpu" not in environment

    def test_rejects_unknown_artifact_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown artifact_kind"):
            build_artifact(
                artifact_kind="not_a_real_kind",
                created_at="2026-08-25T12:00:00Z",
                config={},
                raw_samples_ms=[1.0],
            )

    def test_rejects_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_artifact(
                artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
                created_at="2026-08-25T12:00:00Z",
                config={},
                raw_samples_ms=[],
            )

    def test_rejects_negative_samples(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            build_artifact(
                artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
                created_at="2026-08-25T12:00:00Z",
                config={},
                raw_samples_ms=[1.0, -0.5],
            )

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf])
    def test_rejects_non_finite_samples(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            build_artifact(
                artifact_kind=ARTIFACT_KIND_ONLINE_LATENCY,
                created_at="2026-08-25T12:00:00Z",
                config={},
                raw_samples_ms=[1.0, bad_value],
            )


class TestRoundTrip:
    def test_write_then_read_then_validate_is_clean(self, tmp_path: Path) -> None:
        artifact = _valid_artifact()
        target = tmp_path / "run.json"
        write_artifact_atomic(artifact, target)

        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert artifact_validation_errors(loaded) == []
        assert loaded["raw_samples_ms"] == artifact["raw_samples_ms"]


class TestAtomicWrite:
    def test_no_stray_temp_file_after_a_successful_write(self, tmp_path: Path) -> None:
        write_artifact_atomic(_valid_artifact(), tmp_path / "run.json")
        names = [p.name for p in tmp_path.iterdir()]
        assert names == ["run.json"]

    def test_failed_replace_does_not_touch_an_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "run.json"
        target.write_text('{"already": "here"}', encoding="utf-8")

        def _boom(_src: str, _dst: str) -> None:
            raise OSError("simulated disk failure")

        monkeypatch.setattr("asofline.artifacts.os.replace", _boom)

        with pytest.raises(OSError, match="simulated disk failure"):
            write_artifact_atomic(_valid_artifact(), target)

        assert target.read_text(encoding="utf-8") == '{"already": "here"}'
        leftover = [p for p in tmp_path.iterdir() if p.name != "run.json"]
        assert leftover == []

    def test_writer_refuses_non_finite_values_and_cleans_up(self, tmp_path: Path) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = math.nan  # bypass build_artifact deliberately

        target = tmp_path / "run.json"
        with pytest.raises(ValueError):
            write_artifact_atomic(artifact, target)

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


class TestValidatorHappyPath:
    def test_a_freshly_built_artifact_has_no_errors(self) -> None:
        assert artifact_validation_errors(_valid_artifact()) == []


class TestValidatorTamperDetection:
    """Each test proves the validator catches exactly one kind of doctored file."""

    def test_hand_edited_statistic_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["statistics"]["p99"] = artifact["statistics"]["p99"] + 1_000.0

        errors = artifact_validation_errors(artifact)

        assert any("p99" in e and "does not match" in e for e in errors)

    def test_mutated_raw_sample_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = artifact["raw_samples_ms"][0] + 500.0

        errors = artifact_validation_errors(artifact)

        assert any("sha256" in e for e in errors) or any("statistics" in e for e in errors)

    def test_truncated_raw_samples_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"] = artifact["raw_samples_ms"][:-5]

        errors = artifact_validation_errors(artifact)

        assert any("sha256" in e for e in errors) or any("statistics" in e for e in errors)

    def test_negative_sample_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"].append(-1.0)

        errors = artifact_validation_errors(artifact)

        assert any("negative" in e for e in errors)

    def test_nan_injected_into_raw_samples_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = math.nan

        errors = artifact_validation_errors(artifact)

        assert any("non-finite" in e for e in errors)

    def test_nan_survives_a_permissive_json_round_trip_and_is_still_caught(
        self, tmp_path: Path
    ) -> None:
        """Simulates a file written by something other than this project's own writer.

        ``json.dump`` allows NaN/Infinity by default. This project's own writer refuses
        them (see ``TestAtomicWrite``), but the validator must reject a file containing
        them no matter how it was produced, since it cannot assume every file it is asked
        to check came from ``write_artifact_atomic``.
        """
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = math.nan
        path = tmp_path / "hand_crafted.json"
        path.write_text(json.dumps(artifact, allow_nan=True), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"), parse_constant=float)
        errors = artifact_validation_errors(loaded)

        assert any("non-finite" in e for e in errors)

    def test_infinity_injected_into_raw_samples_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = math.inf

        errors = artifact_validation_errors(artifact)

        assert any("non-finite" in e for e in errors)

    def test_corrupted_hash_with_otherwise_valid_data_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_sha256"] = "0" * 64

        errors = artifact_validation_errors(artifact)

        assert any("sha256" in e for e in errors)
        # Everything else about this artifact is genuinely valid, so the hash mismatch
        # must be the only complaint, proving it is detected independently of the rest.
        assert len(errors) == 1

    @pytest.mark.parametrize("field", [name for name, _ in REQUIRED_FIELDS])
    def test_missing_required_field_is_named_in_the_error(self, field: str) -> None:
        artifact = _valid_artifact()
        del artifact[field]

        errors = artifact_validation_errors(artifact)

        assert any(field in e and "missing" in e for e in errors)

    def test_wrong_schema_version_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["schema_version"] = 999

        errors = artifact_validation_errors(artifact)

        assert any("schema_version" in e for e in errors)

    def test_unknown_artifact_kind_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["artifact_kind"] = "made_up_kind"

        errors = artifact_validation_errors(artifact)

        assert any("artifact_kind" in e for e in errors)

    def test_wrong_type_for_a_required_field_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["config"] = "not a dict"

        errors = artifact_validation_errors(artifact)

        assert any("config" in e and "wrong type" in e for e in errors)

    def test_empty_raw_samples_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"] = []

        errors = artifact_validation_errors(artifact)

        assert any("empty" in e for e in errors)

    def test_non_numeric_raw_sample_is_caught(self) -> None:
        artifact = _valid_artifact()
        artifact["raw_samples_ms"][0] = "12.5"

        errors = artifact_validation_errors(artifact)

        assert any("non-numeric" in e for e in errors)
