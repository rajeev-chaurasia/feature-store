"""Spark session construction, and the Java version gate in front of it.

The gate exists because this machine has four JDKs installed (11, 17, 24 and the
unversioned latest) and Spark 4.0 accepts only 17 and 21. Inheriting whichever one is on
PATH produces a JVM crash several seconds into startup with a stack trace that does not
mention Java versions at all. Asserting up front turns that into one clear sentence.

``/usr/libexec/java_home`` is not consulted: it does not see Homebrew JDKs on this
machine, so it would report "no Java runtime" while four are installed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from asofline.config import SETTINGS, Settings

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_LOG4J_CONFIG = Path(__file__).resolve().parents[3] / "conf" / "log4j2.properties"

ICEBERG_VERSION = "1.11.0"
SUPPORTED_JAVA_MAJORS = frozenset({17, 21})

ICEBERG_PACKAGES = (
    f"org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:{ICEBERG_VERSION}",
    # The AWS bundle carries the SDK that S3FileIO needs. Using it instead of hadoop-aws
    # is what keeps this stack free of hadoop/aws-sdk version conflicts against MinIO.
    f"org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}",
)


class JavaVersionError(RuntimeError):
    """JAVA_HOME points at a JDK this Spark cannot run on."""


def java_major_version(java_home: str) -> int:
    binary = Path(java_home) / "bin" / "java"
    if not binary.is_file():
        raise JavaVersionError(f"no java binary at {binary}")
    result = subprocess.run([str(binary), "-version"], capture_output=True, text=True, check=False)
    # `java -version` writes to stderr on every JDK that matters here.
    banner = result.stderr or result.stdout
    match = re.search(r'version "(\d+)', banner)
    if not match:
        raise JavaVersionError(f"could not parse a version from {banner!r}")
    return int(match.group(1))


def assert_supported_java(settings: Settings = SETTINGS) -> int:
    major = java_major_version(settings.java_home)
    if major not in SUPPORTED_JAVA_MAJORS:
        raise JavaVersionError(
            f"JAVA_HOME={settings.java_home} is Java {major}, but Spark 4.0 supports "
            f"{sorted(SUPPORTED_JAVA_MAJORS)}. Set JAVA_HOME from .env.example."
        )
    return major


def catalog_options(settings: Settings = SETTINGS) -> dict[str, str]:
    """Every ``spark.sql.catalog.<name>.*`` key, in one place.

    Kept separate from the builder so a test can assert the wiring without paying for a
    JVM start.
    """
    prefix = f"spark.sql.catalog.{settings.catalog_name}"
    return {
        prefix: "org.apache.iceberg.spark.SparkCatalog",
        f"{prefix}.type": "rest",
        f"{prefix}.uri": settings.catalog_uri,
        f"{prefix}.warehouse": settings.warehouse,
        f"{prefix}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
        f"{prefix}.s3.endpoint": settings.s3_endpoint,
        f"{prefix}.s3.path-style-access": "true",
        f"{prefix}.s3.access-key-id": settings.s3_access_key,
        f"{prefix}.s3.secret-access-key": settings.s3_secret_key,
        f"{prefix}.client.region": settings.s3_region,
    }


def build_session(
    app_name: str = "asofline",
    *,
    settings: Settings = SETTINGS,
    driver_memory: str = "4g",
    shuffle_partitions: int = 16,
    extra: dict[str, str] | None = None,
) -> SparkSession:
    """A local Spark session wired to the REST catalog.

    ``shuffle_partitions`` defaults far below Spark's 200 because every dataset here fits
    on one laptop; 200 partitions over a few million rows spends all its time on task
    overhead.
    """
    assert_supported_java(settings)
    os.environ["JAVA_HOME"] = settings.java_home

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", ",".join(ICEBERG_PACKAGES))
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.defaultCatalog", settings.catalog_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Timestamps are epoch milliseconds everywhere in this project, and session time
        # zone leaking into a cast is the classic way an as-of join goes quietly wrong.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        # Plain text logs. See conf/log4j2.properties for why the SparkConf flag alone
        # does not achieve this.
        .config("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile={_LOG4J_CONFIG}")
    )
    for key, value in catalog_options(settings).items():
        builder = builder.config(key, value)
    for key, value in (extra or {}).items():
        builder = builder.config(key, value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session
