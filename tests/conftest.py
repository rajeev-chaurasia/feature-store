"""Shared fixtures.

Nothing here is imported by ``tests/unit``. Those tests must run with no JVM, no
containers and no network, so the Spark fixture is built lazily and only the tests that
ask for it pay for it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    from asofline.offline.session import build_session

    session = build_session("asofline-tests", driver_memory="2g", shuffle_partitions=4)
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture(scope="session")
def test_namespace(spark: SparkSession) -> Iterator[str]:
    namespace = "asofline_test"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
    yield namespace
