"""Feature definitions.

This package and ``asofline.agg`` are the two that must not import from ``offline``,
``online``, ``streaming`` or ``serving``. ``tests/unit/test_layering.py`` enforces it.
The reason is that the shared window semantics have to be testable with no JVM, no
containers and no network, in the same way that warpline's correctness gate imports no
torch.
"""

from asofline.definitions.aggregation import AggFunction, Aggregation, format_window
from asofline.definitions.entity import Entity, KeyType
from asofline.definitions.errors import DefinitionError
from asofline.definitions.registry import Registry
from asofline.definitions.resolution import FIVE_MINUTE_RESOLUTION, Resolution
from asofline.definitions.source import EventSource
from asofline.definitions.view import FeatureView

__all__ = [
    "FIVE_MINUTE_RESOLUTION",
    "AggFunction",
    "Aggregation",
    "DefinitionError",
    "Entity",
    "EventSource",
    "FeatureView",
    "KeyType",
    "Registry",
    "Resolution",
    "format_window",
]
