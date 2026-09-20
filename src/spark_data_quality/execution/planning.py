"""Translate expectation definitions into one Spark aggregation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    IntegralType,
    MapType,
    NumericType,
    StringType,
    StructField,
)

from spark_data_quality.checks import (
    AllowedValues,
    Check,
    ColumnExists,
    HasType,
    InRange,
    MatchesRegex,
    NotNull,
    RowCount,
    Unique,
)
from spark_data_quality.models import Diagnostic

MetricKey = tuple[str, str | None, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class Metric:
    """A named scalar Spark aggregation expression."""

    key: MetricKey
    alias: str
    expression: Column


@dataclass(frozen=True, slots=True)
class PreparedCheck:
    """One check resolved against a DataFrame schema."""

    check: Check
    check_id: str
    metric_aliases: dict[str, str] = field(default_factory=dict)
    schema_observation: Any | None = None
    diagnostic: Diagnostic | None = None
    planning_duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AggregationPlan:
    """All checks plus their deduplicated aggregate expressions."""

    checks: tuple[PreparedCheck, ...]
    metrics: tuple[Metric, ...]
    schema_check_count: int


class AggregationPlanner:
    """Build a deterministic, shared aggregation for compatible checks."""

    def __init__(self, dataframe: DataFrame) -> None:
        self._dataframe = dataframe
        self._metrics: dict[MetricKey, Metric] = {}

    def plan(self, checks: tuple[Check, ...]) -> AggregationPlan:
        prepared = tuple(self._prepare(check, index) for index, check in enumerate(checks))
        schema_checks = sum(isinstance(item.check, (ColumnExists, HasType)) for item in prepared)
        return AggregationPlan(
            checks=prepared,
            metrics=tuple(self._metrics.values()),
            schema_check_count=schema_checks,
        )

    def _prepare(self, check: Check, index: int) -> PreparedCheck:
        started = perf_counter()
        check_id = check.check_id or f"{check.check_type}:{index}"
        missing = [name for name in check.required_columns if name not in self._dataframe.columns]
        if missing:
            if isinstance(check, ColumnExists):
                return PreparedCheck(
                    check=check,
                    check_id=check_id,
                    schema_observation=False,
                    planning_duration_ms=_elapsed_ms(started),
                )
            return PreparedCheck(
                check=check,
                check_id=check_id,
                diagnostic=Diagnostic(
                    code="MISSING_COLUMN",
                    message=f"Required column {missing[0]!r} does not exist",
                    column=missing[0],
                ),
                planning_duration_ms=_elapsed_ms(started),
            )

        if isinstance(check, ColumnExists):
            return PreparedCheck(
                check=check,
                check_id=check_id,
                schema_observation=True,
                planning_duration_ms=_elapsed_ms(started),
            )

        if isinstance(check, HasType):
            actual_type = self._field(check.column).dataType
            return PreparedCheck(
                check=check,
                check_id=check_id,
                schema_observation=actual_type.simpleString(),
                planning_duration_ms=_elapsed_ms(started),
            )

        type_error = self._validate_spark_type(check)
        if type_error is not None:
            return PreparedCheck(
                check=check,
                check_id=check_id,
                diagnostic=type_error,
                planning_duration_ms=_elapsed_ms(started),
            )

        aliases: dict[str, str] = {}
        aliases["total"] = self._metric(("total", None, ()), F.count(F.lit(1)))

        if isinstance(check, NotNull):
            aliases["invalid"] = self._metric(
                ("null_count", check.column, ()),
                F.sum(F.when(_column(check.column).isNull(), 1).otherwise(0)),
            )
        elif isinstance(check, Unique):
            aliases["non_null"] = self._metric(
                ("non_null_count", check.column, ()), F.count(_column(check.column))
            )
            aliases["distinct"] = self._metric(
                ("distinct_count", check.column, ()), F.countDistinct(_column(check.column))
            )
        elif isinstance(check, InRange):
            invalid = _range_invalid_expression(check)
            aliases["invalid"] = self._metric(
                ("range_invalid", check.column, _range_args(check)),
                _invalid_count(check.column, invalid),
            )
        elif isinstance(check, AllowedValues):
            ordered_values = tuple(
                sorted(check.values, key=lambda value: (type(value).__name__, str(value)))
            )
            aliases["invalid"] = self._metric(
                ("allowed_invalid", check.column, ordered_values),
                _invalid_count(check.column, ~_column(check.column).isin(*ordered_values)),
            )
        elif isinstance(check, MatchesRegex):
            aliases["invalid"] = self._metric(
                ("regex_invalid", check.column, (check.pattern,)),
                _invalid_count(check.column, ~_column(check.column).rlike(check.pattern)),
            )
        elif not isinstance(check, RowCount):
            raise TypeError(f"Unsupported check type: {type(check).__name__}")

        return PreparedCheck(
            check=check,
            check_id=check_id,
            metric_aliases=aliases,
            planning_duration_ms=_elapsed_ms(started),
        )

    def _metric(self, key: MetricKey, expression: Column) -> str:
        existing = self._metrics.get(key)
        if existing is not None:
            return existing.alias
        alias = f"metric_{len(self._metrics):04d}"
        self._metrics[key] = Metric(key=key, alias=alias, expression=expression.alias(alias))
        return alias

    def _field(self, column: str) -> StructField:
        return self._dataframe.schema[column]

    def _validate_spark_type(self, check: Check) -> Diagnostic | None:
        if not check.required_columns:
            return None
        column = check.required_columns[0]
        actual = self._field(column).dataType
        expected: str | None = None

        if isinstance(check, InRange) and not isinstance(actual, NumericType):
            expected = "numeric"
        elif isinstance(check, MatchesRegex) and not isinstance(actual, StringType):
            expected = "string"
        elif isinstance(check, AllowedValues):
            value = next(iter(check.values))
            compatible = (
                (isinstance(value, str) and isinstance(actual, StringType))
                or (isinstance(value, bool) and isinstance(actual, BooleanType))
                or (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and isinstance(actual, IntegralType)
                )
                or (isinstance(value, float) and isinstance(actual, NumericType))
            )
            if not compatible:
                expected = f"compatible with {type(value).__name__} allowed values"
        elif isinstance(check, Unique) and isinstance(actual, MapType):
            expected = "a Spark type that supports exact distinct aggregation"

        if expected is None:
            return None
        return Diagnostic(
            code="UNEXPECTED_SPARK_TYPE",
            message=(
                f"Column {column!r} has Spark type {actual.simpleString()!r}; expected {expected}"
            ),
            column=column,
            details={"actual_type": actual.simpleString(), "expected_type": expected},
        )


def _column(name: str) -> Column:
    escaped = name.replace("`", "``")
    return F.col(f"`{escaped}`")


def _invalid_count(column: str, invalid: Column) -> Column:
    return F.sum(F.when(_column(column).isNotNull() & invalid, 1).otherwise(0))


def _range_invalid_expression(check: InRange) -> Column:
    value = _column(check.column)
    invalid = F.lit(False)
    if check.min_value is not None:
        below_minimum = value < check.min_value if check.inclusive else value <= check.min_value
        invalid = invalid | below_minimum
    if check.max_value is not None:
        above_maximum = value > check.max_value if check.inclusive else value >= check.max_value
        invalid = invalid | above_maximum
    return invalid


def _range_args(check: InRange) -> tuple[Any, ...]:
    return (check.min_value, check.max_value, check.inclusive)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000
