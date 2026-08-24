"""Errors raised while a feature definition is being constructed.

Definitions validate eagerly, at construction time, so an unusable registry cannot
reach a Spark job or a Kafka consumer and fail there instead.
"""

from __future__ import annotations


class DefinitionError(ValueError):
    """A feature definition is internally inconsistent or unsupported."""
