"""Built-in expectation definitions."""

from __future__ import annotations

import math
from collections.abc import Set
from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar

from pyspark.sql.types import DataType

from spark_data_quality.checks.base import Check, require_column_name
from spark_data_quality.models.values import JsonValue

Scalar = str | int | float | bool | Decimal
RangeBound = int | float | Decimal


@dataclass(frozen=True, slots=True)
class ColumnCheck(Check):
    """Base for checks configured with one column."""

    column: str

    def __post_init__(self) -> None:
        Check.__post_init__(self)
        require_column_name(self.column)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)


@dataclass(frozen=True, slots=True)
class ColumnExists(ColumnCheck):
    check_type: ClassVar[str] = "column_exists"

    @property
    def description(self) -> str:
        return f"Column {self.column!r} exists"

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column, "exists": True}


@dataclass(frozen=True, slots=True)
class HasType(ColumnCheck):
    expected_type: DataType
    check_type: ClassVar[str] = "has_type"

    def __post_init__(self) -> None:
        ColumnCheck.__post_init__(self)
        if not isinstance(self.expected_type, DataType) or type(self.expected_type) is DataType:
            raise ValueError("expected_type must be a concrete Spark DataType instance")

    @property
    def description(self) -> str:
        return f"Column {self.column!r} has Spark type {self.expected_type.simpleString()!r}"

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column, "spark_type": self.expected_type.simpleString()}


@dataclass(frozen=True, slots=True)
class NotNull(ColumnCheck):
    check_type: ClassVar[str] = "not_null"

    @property
    def description(self) -> str:
        return f"Column {self.column!r} contains no null values"

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column, "null_count": 0}


@dataclass(frozen=True, slots=True)
class Unique(ColumnCheck):
    check_type: ClassVar[str] = "unique"

    @property
    def description(self) -> str:
        return f"Non-null values in column {self.column!r} are unique"

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column, "duplicate_excess_count": 0, "nulls": "ignored"}


@dataclass(frozen=True, slots=True)
class InRange(ColumnCheck):
    min_value: RangeBound | None = None
    max_value: RangeBound | None = None
    inclusive: bool = True
    check_type: ClassVar[str] = "in_range"

    def __post_init__(self) -> None:
        ColumnCheck.__post_init__(self)
        if self.min_value is None and self.max_value is None:
            raise ValueError("at least one of min_value and max_value is required")
        for name, value in (("min_value", self.min_value), ("max_value", self.max_value)):
            if value is not None:
                _validate_range_bound(name, value)
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must not exceed max_value")

    @property
    def description(self) -> str:
        boundary = "inclusive" if self.inclusive else "exclusive"
        return (
            f"Non-null values in column {self.column!r} are within the configured {boundary} range"
        )

    def expected(self) -> dict[str, JsonValue]:
        return {
            "column": self.column,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive,
            "nulls": "ignored",
        }


@dataclass(frozen=True, slots=True)
class AllowedValues(ColumnCheck):
    values: Set[Scalar] = field(default_factory=frozenset)
    check_type: ClassVar[str] = "allowed_values"

    def __post_init__(self) -> None:
        ColumnCheck.__post_init__(self)
        object.__setattr__(self, "values", frozenset(self.values))
        if not self.values:
            raise ValueError("values must contain at least one allowed value")
        value_types = {type(value) for value in self.values}
        if len(value_types) != 1 or not value_types <= {str, int, float, bool, Decimal}:
            raise ValueError("allowed values must all have the same scalar type")
        if any(
            (isinstance(value, float) and not math.isfinite(value))
            or (isinstance(value, Decimal) and not value.is_finite())
            for value in self.values
        ):
            raise ValueError("allowed numeric values must be finite")

    @property
    def description(self) -> str:
        return f"Non-null values in column {self.column!r} belong to the allowed set"

    def expected(self) -> dict[str, JsonValue]:
        ordered: list[JsonValue] = sorted(
            self.values, key=lambda value: (type(value).__name__, str(value))
        )
        return {"column": self.column, "allowed_values": ordered, "nulls": "ignored"}


@dataclass(frozen=True, slots=True)
class MatchesRegex(ColumnCheck):
    pattern: str = ""
    check_type: ClassVar[str] = "matches_regex"

    def __post_init__(self) -> None:
        ColumnCheck.__post_init__(self)
        if not self.pattern:
            raise ValueError("pattern must be non-empty")

    @property
    def description(self) -> str:
        return f"Non-null values in column {self.column!r} match the configured pattern"

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column, "pattern": self.pattern, "nulls": "ignored"}


@dataclass(frozen=True, slots=True)
class RowCount(Check):
    min_count: int | None = None
    max_count: int | None = None
    check_type: ClassVar[str] = "row_count"

    def __post_init__(self) -> None:
        Check.__post_init__(self)
        if self.min_count is None and self.max_count is None:
            raise ValueError("at least one of min_count and max_count is required")
        for name, value in (("min_count", self.min_count), ("max_count", self.max_count)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer")
        if self.min_count is not None and self.min_count < 0:
            raise ValueError("min_count must be non-negative")
        if self.max_count is not None and self.max_count < 0:
            raise ValueError("max_count must be non-negative")
        if (
            self.min_count is not None
            and self.max_count is not None
            and self.min_count > self.max_count
        ):
            raise ValueError("min_count must not exceed max_count")

    @property
    def description(self) -> str:
        return "DataFrame row count satisfies the configured bounds"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ()

    def expected(self) -> dict[str, JsonValue]:
        return {"min_count": self.min_count, "max_count": self.max_count}


def _validate_range_bound(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    finite = value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
    if not finite:
        raise ValueError(f"{name} must be a finite number")
