"""One naming rule, applied everywhere.

A view name becomes an Iceberg table name, a Redis key segment and a JSON field in the
serving response. Those three tolerate different character sets, so the definition layer
takes the intersection and refuses anything else rather than letting the disagreement
surface as a runtime error in whichever layer is least forgiving.
"""

from __future__ import annotations

import re

from asofline.definitions.errors import DefinitionError

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_identifier(value: str, *, kind: str) -> str:
    if not _IDENTIFIER.match(value):
        raise DefinitionError(
            f"{kind} {value!r} must be lower snake case matching {_IDENTIFIER.pattern}"
        )
    return value
