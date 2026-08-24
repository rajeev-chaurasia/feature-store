"""Entities: the things features are attached to."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from asofline.definitions.naming import validate_identifier


class KeyType(StrEnum):
    STRING = "string"
    INT64 = "int64"


@dataclass(frozen=True, slots=True)
class Entity:
    """A join key plus its type.

    ``join_key`` is separate from ``name`` because the column in the event stream is not
    always the name you want in a feature view, and coupling them makes renaming either
    one a breaking change to the other.
    """

    name: str
    join_key: str
    key_type: KeyType = KeyType.STRING
    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.name, kind="entity name")
        validate_identifier(self.join_key, kind="join key")
