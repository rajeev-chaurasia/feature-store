"""CI's entry point for re-validating committed evidence artifacts.

Deliberately thin. Every validation rule lives in
``asofline.artifacts.artifact_validation_errors``; this script's only job is to find
files, parse them, and turn a list of error strings into a pass/fail exit code.

    uv run python scripts/validate_artifacts.py results/2026-08-25-p3-online/run.json
    uv run python scripts/validate_artifacts.py results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asofline.artifacts import artifact_validation_errors


def _artifact_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.rglob("*.json"))
    return [target]


def _validate_one(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return [f"could not read file: {error}"]

    try:
        artifact = json.loads(text)
    except json.JSONDecodeError as error:
        return [f"not valid JSON: {error}"]

    if not isinstance(artifact, dict):
        return ["top-level JSON value is not an object"]

    return artifact_validation_errors(artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="an artifact JSON file, or a directory to scan recursively for *.json files",
    )
    args = parser.parse_args(argv)

    target: Path = args.target
    if not target.exists():
        print(f"FAIL {target}: path does not exist", file=sys.stderr)
        return 1

    files = _artifact_files(target)
    if not files:
        print(f"no *.json files found under {target}", file=sys.stderr)
        return 1

    all_passed = True
    for path in files:
        errors = _validate_one(path)
        if errors:
            all_passed = False
            print(f"FAIL {path}")
            for error in errors:
                print(f"    {error}")
        else:
            print(f"PASS {path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
