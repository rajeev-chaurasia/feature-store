"""Subprocess entry point: run the real Kafka-to-Redis consumer for a freshness probe.

Invoked as ``python -m asofline.bench._freshness_consumer_worker`` with its Kafka
bootstrap, topic, Redis URL and consumer group passed on the command line, so the probe
harness controls exactly which instance of the pipeline it is timing without touching
``asofline.streaming.consumer`` itself. This runs the exact same ``consumer.run()`` loop a
real deployment uses; the only difference from that module's own ``__main__`` block is
that every endpoint is overridable rather than fixed to ``SETTINGS``, so a probe can point
a disposable instance at a scratch Redis database and a run-scoped consumer group without
colliding with a production consumer using the real ``asofline-to-redis`` group.

Shutdown is a plain ``SIGINT``: Python's default handler turns that into a
``KeyboardInterrupt`` in this process's main thread, which unwinds through
``consumer.run()``'s own ``finally`` clause and calls ``Consumer.close()``, leaving the
consumer group cleanly. See ``asofline.bench.freshness.stop_consumer_subprocess`` for the
harness side of this.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging

from asofline.config import SETTINGS
from asofline.demo.views import DEMO_REGISTRY
from asofline.streaming import consumer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kafka-bootstrap", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--group-id", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    settings = dataclasses.replace(
        SETTINGS,
        kafka_bootstrap=args.kafka_bootstrap,
        events_topic=args.topic,
        redis_url=args.redis_url,
    )
    kafka_consumer = consumer.build_consumer(settings=settings, group_id=args.group_id)
    redis_client = consumer.build_redis_client(settings=settings)
    try:
        consumer.run(kafka_consumer, redis_client, DEMO_REGISTRY, topic=settings.events_topic)
    finally:
        redis_client.close()


if __name__ == "__main__":  # pragma: no cover
    # The expected shutdown path (see the module docstring): SIGINT from the harness
    # unwinds through consumer.run()'s finally clause, then arrives here. A traceback for
    # an intentional, harness-requested stop would just be log noise.
    with contextlib.suppress(KeyboardInterrupt):
        main()
