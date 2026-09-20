"""Typed expectation definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True, kw_only=True)
class Check(ABC):
    """An immutable data-quality expectation.

    Checks contain configuration only. Spark planning and result interpretation
    live in the execution package.
    """

    check_id: str | None = None
    check_type: ClassVar[str]

    def __post_init__(self) -> None:
        if self.check_id is not None and not self.check_id.strip():
            raise ValueError("check_id must be non-empty when provided")

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable expectation description."""

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Return columns required to evaluate this check."""

    @abstractmethod
    def expected(self) -> dict[str, Any]:
        """Return a JSON-compatible expected constraint."""


def require_column_name(column: str) -> None:
    """Validate a public column-name argument."""

    if not column or not column.strip():
        raise ValueError("column must be a non-empty string")
