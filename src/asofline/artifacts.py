"""The evidence-artifact schema shared by the online store and streaming benchmarks.

Both P3 (Redis online-store latency) and P4 (streaming freshness/lag) publish the same
shape of claim: "here is a distribution of latency measurements, and here is what I say
about it." A benchmark that prints only a p99 is unfalsifiable from outside the process
that produced it. The fix, following warpline, is to never let the summary outrun the
evidence: ``raw_samples_ms`` is committed in full, ``statistics`` is always derived from
it by this module rather than handed in by the caller, and a sha256 over the raw samples
lets a hand-edited statistics block (or a hand-edited sample list) be caught even before
anyone bothers to recompute anything.

One schema, not two, because ``artifact_kind`` is the only thing that differs between an
online-latency run and a streaming-lag run: both are "a bag of millisecond measurements
plus the config that produced them." Splitting the schema would duplicate the validator,
and a duplicated validator is the one thing this module exists to prevent.

**Percentile convention.** Percentiles use linear interpolation between closest ranks
(``numpy``'s historical default, sometimes called the "linear" or "R-7" method):

    rank = p * (n - 1)

and the value is interpolated between ``sorted[floor(rank)]`` and ``sorted[ceil(rank)]``.
This is the convention where, for an even-length sample, the p50 exactly matches the
textbook definition of the median (interpolate the two middle values), so the ``median``
field and the ``p50`` computation never have two independent implementations that could
drift apart.

**Timestamps and environment are supplied by the caller, not sampled internally.**
``build_artifact`` never calls ``datetime.now()``: it takes ``created_at`` as an already
formatted ISO8601 string. A function that reads the clock cannot be tested for a fixed
answer, and every other part of this module is designed to have one.

**The writer refuses non-finite samples.** ``json`` can round-trip NaN and Infinity by
default (``allow_nan=True``), which would make "the file is valid JSON" a weaker claim
than "the file is a valid artifact." This module writes with ``allow_nan=False`` and
rejects non-finite or negative samples before an artifact is even built, so a NaN or a
negative latency can only appear in a committed file via direct tampering after the fact,
which is exactly the class of file ``artifact_validation_errors`` exists to catch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

ARTIFACT_KIND_ONLINE_LATENCY = "online_latency"
ARTIFACT_KIND_STREAMING_LAG = "streaming_lag"
KNOWN_ARTIFACT_KINDS = frozenset({ARTIFACT_KIND_ONLINE_LATENCY, ARTIFACT_KIND_STREAMING_LAG})

STATISTIC_FIELDS = ("count", "min", "max", "mean", "median", "p90", "p99", "mad")

# (name, expected type) for every top-level key an artifact must carry. Kept as a single
# table so the validator has exactly one place that knows the shape of the schema.
REQUIRED_FIELDS: tuple[tuple[str, type], ...] = (
    ("schema_version", int),
    ("artifact_kind", str),
    ("created_at", str),
    ("config", dict),
    ("raw_samples_ms", list),
    ("raw_samples_sha256", str),
    ("statistics", dict),
    ("environment", dict),
)

_STAT_TOLERANCE_REL = 1e-9
_STAT_TOLERANCE_ABS = 1e-9


def _is_number(value: object) -> bool:
    """True for a real ``int``/``float``, excluding ``bool`` (a ``bool`` is an ``int``)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear interpolation between closest ranks. ``sorted_values`` must already be sorted."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = fraction * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def compute_statistics(raw_samples_ms: list[float]) -> dict[str, float | int]:
    """Recompute every summary statistic from ``raw_samples_ms``. Pure, no I/O.

    This is the only function in the project allowed to produce a ``statistics`` block.
    Both ``build_artifact`` (trusted, freshly measured samples) and
    ``artifact_validation_errors`` (untrusted, possibly tampered samples) call it, so
    there is exactly one definition of what "correct statistics" means.
    """
    if not raw_samples_ms:
        raise ValueError("cannot compute statistics over an empty sample list")
    values = [float(v) for v in raw_samples_ms]
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"cannot compute statistics: non-finite sample {value!r}")

    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    median = _percentile(ordered, 0.5)
    deviations = sorted(abs(value - median) for value in ordered)

    return {
        "count": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": median,
        "p90": _percentile(ordered, 0.9),
        "p99": _percentile(ordered, 0.99),
        "mad": _percentile(deviations, 0.5),
    }


def _hash_samples(raw_samples_ms: list[float]) -> str:
    """sha256 over the raw sample array, computed from the in-memory values.

    Hashing the parsed values rather than the file's raw bytes means the hash is
    insensitive to formatting (key order, whitespace, trailing zeros in the JSON text)
    and only reflects the measurements themselves, which is the thing it is meant to
    protect. ``allow_nan=True`` here is deliberate: hashing must be able to run over a
    tampered sample list that contains NaN, so that the sha256 check and the finiteness
    check are independent detectors rather than one masking the other.
    """
    payload = json.dumps(raw_samples_ms, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_environment() -> dict[str, Any]:
    """A minimal, honest snapshot of the machine a run executed on.

    This project runs on one dev machine with no GPU and no cluster, so the fields are
    limited to what is actually true of that machine: interpreter version, OS platform,
    hostname, CPU count, and whether the process is inside a container (relevant because
    P3/P4 compare in-container Redis/Kafka against a host process). Fabricating fields
    that do not apply here (accelerator counts, node counts) would make the environment
    block look more rigorous than it is.
    """
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "in_docker": Path("/.dockerenv").exists(),
    }


def build_artifact(
    *,
    artifact_kind: str,
    created_at: str,
    config: dict[str, Any],
    raw_samples_ms: list[float],
    schema_version: int = SCHEMA_VERSION,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a full artifact dict from raw samples plus the config that produced them.

    ``statistics`` is always computed here, from ``raw_samples_ms``, never accepted as an
    argument: a caller that could hand in its own statistics block could hand in a wrong
    one, and that is precisely the failure mode this module exists to make impossible by
    construction rather than catch after the fact.
    """
    if artifact_kind not in KNOWN_ARTIFACT_KINDS:
        raise ValueError(
            f"unknown artifact_kind {artifact_kind!r}; expected one of "
            f"{sorted(KNOWN_ARTIFACT_KINDS)}"
        )
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; supported versions are "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if not raw_samples_ms:
        raise ValueError(
            "raw_samples_ms must be non-empty; a run with no measurements is not evidence"
        )

    values = [float(v) for v in raw_samples_ms]
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"raw_samples_ms contains a non-finite value: {value!r}")
        if value < 0:
            raise ValueError(
                f"raw_samples_ms contains a negative value: {value!r}; latency cannot be negative"
            )

    return {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "created_at": created_at,
        "config": config,
        "raw_samples_ms": values,
        "raw_samples_sha256": _hash_samples(values),
        "statistics": compute_statistics(values),
        "environment": environment if environment is not None else capture_environment(),
    }


def write_artifact_atomic(artifact: dict[str, Any], path: Path) -> None:
    """Write ``artifact`` to ``path`` such that no concurrent reader ever sees a partial file.

    A temp file in the same directory as ``path`` guarantees ``os.replace`` is a rename on
    one filesystem, which POSIX makes atomic. If anything fails before the replace (a
    serialization error, a full disk, a mocked failure in tests) the temp file is removed
    and ``path`` is left exactly as it was.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=directory,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_name = tmp.name
            # allow_nan=False: a NaN or Infinity here means build_artifact was bypassed,
            # and the writer should fail loudly rather than emit a file that looks like
            # JSON but is not a valid artifact by this project's own rules.
            json.dump(artifact, tmp, indent=2, sort_keys=True, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _statistics_mismatch_errors(
    stored: dict[str, Any], recomputed: dict[str, float | int]
) -> list[str]:
    errors: list[str] = []
    for field in STATISTIC_FIELDS:
        if field not in stored:
            errors.append(f"statistics is missing required field: {field!r}")
            continue
        value = stored[field]
        if not _is_number(value):
            errors.append(f"statistics[{field!r}] = {value!r} is not numeric")
            continue
        expected = recomputed[field]
        if not math.isclose(
            float(value), float(expected), rel_tol=_STAT_TOLERANCE_REL, abs_tol=_STAT_TOLERANCE_ABS
        ):
            errors.append(
                f"statistics[{field!r}] = {value!r} does not match the value recomputed "
                f"from raw_samples_ms ({expected!r})"
            )
    return errors


def artifact_validation_errors(artifact: dict[str, Any]) -> list[str]:
    """Every reason ``artifact`` should not be trusted. Empty list means it is valid.

    This is the single place all validation logic lives; ``scripts/validate_artifacts.py``
    and every tamper test call this function rather than reimplementing any piece of it.
    Recomputation, not trust, is the point: statistics and the sha256 are both derived
    again from ``raw_samples_ms`` and compared to what is stored, so a file can only pass
    by actually being internally consistent.
    """
    if not isinstance(artifact, dict):
        return ["artifact is not a JSON object"]

    errors: list[str] = []

    for field, expected_type in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field!r}")
            continue
        actual = artifact[field]
        wrong_type = not isinstance(actual, expected_type) or (
            expected_type is int and isinstance(actual, bool)
        )
        if wrong_type:
            errors.append(
                f"field {field!r} has wrong type: expected {expected_type.__name__}, "
                f"got {type(actual).__name__}"
            )

    schema_version = artifact.get("schema_version")
    is_plain_int = isinstance(schema_version, int) and not isinstance(schema_version, bool)
    if is_plain_int and schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"unsupported schema_version {schema_version!r}; supported versions are "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    artifact_kind = artifact.get("artifact_kind")
    if isinstance(artifact_kind, str) and artifact_kind not in KNOWN_ARTIFACT_KINDS:
        errors.append(
            f"unknown artifact_kind {artifact_kind!r}; known kinds are "
            f"{sorted(KNOWN_ARTIFACT_KINDS)}"
        )

    raw_samples = artifact.get("raw_samples_ms")
    if isinstance(raw_samples, list):
        errors.extend(_raw_samples_errors(raw_samples, artifact))

    return errors


def _raw_samples_errors(raw_samples: list[Any], artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not raw_samples:
        errors.append("raw_samples_ms is empty; a run with no measurements is not evidence")
        return errors

    non_numeric = [v for v in raw_samples if not _is_number(v)]
    if non_numeric:
        errors.append(f"raw_samples_ms contains {len(non_numeric)} non-numeric value(s)")
        return errors

    numeric_samples = [float(v) for v in raw_samples]

    non_finite_count = sum(1 for v in numeric_samples if not math.isfinite(v))
    if non_finite_count:
        errors.append(
            f"raw_samples_ms contains {non_finite_count} non-finite value(s) (nan or "
            "infinity); a latency measurement must be finite"
        )

    negative_count = sum(1 for v in numeric_samples if v < 0)
    if negative_count:
        errors.append(
            f"raw_samples_ms contains {negative_count} negative value(s); "
            "a latency cannot be negative"
        )

    stored_hash = artifact.get("raw_samples_sha256")
    if isinstance(stored_hash, str):
        recomputed_hash = _hash_samples(numeric_samples)
        if recomputed_hash != stored_hash:
            errors.append(
                "raw_samples_sha256 does not match the sha256 recomputed from "
                f"raw_samples_ms (stored={stored_hash!r}, recomputed={recomputed_hash!r})"
            )

    stored_statistics = artifact.get("statistics")
    if not non_finite_count and isinstance(stored_statistics, dict):
        recomputed_statistics = compute_statistics(numeric_samples)
        errors.extend(_statistics_mismatch_errors(stored_statistics, recomputed_statistics))

    return errors
