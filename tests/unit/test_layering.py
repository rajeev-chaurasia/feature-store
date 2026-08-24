"""The one-way dependency rule, enforced rather than documented.

``definitions`` and ``agg`` hold the window semantics that both the batch path and the
streaming path obey. If either of them grows an import of Spark, Redis or Kafka, the
semantics can no longer be tested without that dependency, and in practice that means
they stop being tested at the boundaries where they are most likely to be wrong.

This is the same discipline as warpline's correctness gate, which imports no torch and so
runs on a machine with no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "asofline"
PURE_PACKAGES = ("definitions", "agg")
FORBIDDEN_PREFIXES = (
    "asofline.offline",
    "asofline.online",
    "asofline.streaming",
    "asofline.serving",
    "asofline.skew",
    "asofline.bench",
    "pyspark",
    "redis",
    "confluent_kafka",
    "fastapi",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _python_files(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("package", PURE_PACKAGES)
def test_pure_packages_have_files(package: str) -> None:
    assert _python_files(package), f"{package} has no modules, so the rule is vacuous"


@pytest.mark.parametrize("package", PURE_PACKAGES)
def test_pure_packages_import_no_infrastructure(package: str) -> None:
    offences: list[str] = []
    for path in _python_files(package):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES):
                offences.append(f"{path.relative_to(SOURCE_ROOT)} imports {module}")
    assert not offences, "\n".join(offences)


def test_definitions_import_without_any_optional_dependency() -> None:
    """A bare import must work with only the base install.

    ``importlib`` rather than a module-level import, so the failure names this rule
    instead of collecting as an error in every other test in the file.
    """
    import importlib

    module = importlib.import_module("asofline.definitions")
    assert module.Registry is not None
