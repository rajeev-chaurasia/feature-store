"""Runtime endpoints and credentials, read once from the environment.

Defaults match ``docker-compose.yml`` so a fresh clone runs with no ``.env`` at all. The
defaults are development values; nothing here should ever point at a real account.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    java_home: str
    catalog_uri: str
    catalog_name: str
    warehouse: str
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    kafka_bootstrap: str
    events_topic: str
    feature_log_topic: str
    redis_url: str

    @classmethod
    def from_env(cls) -> Settings:
        get = os.environ.get
        return cls(
            java_home=get(
                "JAVA_HOME", "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
            ),
            catalog_uri=get("ASOFLINE_CATALOG_URI", "http://localhost:8181"),
            catalog_name=get("ASOFLINE_CATALOG_NAME", "asofline"),
            warehouse=get("ASOFLINE_WAREHOUSE", "s3://asofline-warehouse/"),
            s3_endpoint=get("ASOFLINE_S3_ENDPOINT", "http://localhost:9000"),
            s3_access_key=get("AWS_ACCESS_KEY_ID", "asofline"),
            s3_secret_key=get("AWS_SECRET_ACCESS_KEY", "asofline-dev-secret"),
            s3_region=get("AWS_REGION", "us-east-1"),
            kafka_bootstrap=get("ASOFLINE_KAFKA_BOOTSTRAP", "localhost:9092"),
            events_topic=get("ASOFLINE_EVENTS_TOPIC", "engagement_events"),
            feature_log_topic=get("ASOFLINE_FEATURE_LOG_TOPIC", "feature_logs"),
            redis_url=get("ASOFLINE_REDIS_URL", "redis://localhost:6379/0"),
        )


SETTINGS = Settings.from_env()
