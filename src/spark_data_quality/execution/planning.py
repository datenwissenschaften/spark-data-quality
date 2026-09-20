"""Translate expectation definitions into one Spark aggregation plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from pyspark.errors import IllegalArgumentException
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    FloatType,
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

type MetricArgument = None | bool | int | float | str | Decimal
type MetricKey = tuple[str, str | None, tuple[MetricArgument, ...]]


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
    schema_observation: bool | DataType | None = None
    diagnostic: Diagnostic | None = None


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
        check_id = check.check_id or f"{check.check_type}:{index}"
        missing = [name for name in check.required_columns if name not in self._dataframe.columns]
        if missing:
            if isinstance(check, ColumnExists):
                return PreparedCheck(
                    check=check,
                    check_id=check_id,
                    schema_observation=False,
                )
            return PreparedCheck(
                check=check,
                check_id=check_id,
                diagnostic=Diagnostic(
                    code="MISSING_COLUMN",
                    message=f"Required column {missing[0]!r} does not exist",
                    column=missing[0],
                ),
            )

        if isinstance(check, ColumnExists):
            return PreparedCheck(
                check=check,
                check_id=check_id,
                schema_observation=True,
            )

        if isinstance(check, HasType):
            actual_type = self._field(check.column).dataType
            return PreparedCheck(
                check=check,
                check_id=check_id,
                schema_observation=actual_type,
            )

        type_error = self._validate_spark_type(check)
        if type_error is not None:
            return PreparedCheck(
                check=check,
                check_id=check_id,
                diagnostic=type_error,
            )

        aliases: dict[str, str] = {}
        aliases["total"] = self._metric(("total", None, ()), F.count(F.lit(1)))

        if isinstance(check, NotNull):
            aliases["invalid"] = self._metric(
                ("null_count", check.column, ()),
                F.sum(F.when(_column(check.column).isNull(), 1).otherwise(0)),
            )
        elif isinstance(check, Unique):
            aliases["evaluated"] = self._metric(
                ("non_null_count", check.column, ()), F.count(_column(check.column))
            )
            aliases["distinct"] = self._metric(
                ("distinct_count", check.column, ()), F.countDistinct(_column(check.column))
            )
        elif isinstance(check, InRange):
            aliases["evaluated"] = self._metric(
                ("non_null_count", check.column, ()), F.count(_column(check.column))
            )
            invalid = _range_invalid_expression(check, self._field(check.column).dataType)
            aliases["invalid"] = self._metric(
                ("range_invalid", check.column, _range_args(check)),
                _invalid_count(check.column, invalid),
            )
        elif isinstance(check, AllowedValues):
            aliases["evaluated"] = self._metric(
                ("non_null_count", check.column, ()), F.count(_column(check.column))
            )
            ordered_values = tuple(
                sorted(check.values, key=lambda value: (type(value).__name__, str(value)))
            )
            aliases["invalid"] = self._metric(
                ("allowed_invalid", check.column, ordered_values),
                _invalid_count(check.column, ~_column(check.column).isin(*ordered_values)),
            )
        elif isinstance(check, MatchesRegex):
            aliases["evaluated"] = self._metric(
                ("non_null_count", check.column, ()), F.count(_column(check.column))
            )
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

        if isinstance(check, InRange):
            if not isinstance(actual, NumericType):
                expected = "numeric"
            elif isinstance(actual, DecimalType) and any(
                isinstance(bound, float)
                for bound in (check.min_value, check.max_value)
                if bound is not None
            ):
                return _bound_type_diagnostic(
                    column,
                    actual,
                    "Decimal or integer bounds for a decimal column",
                )
            elif not isinstance(actual, DecimalType) and any(
                isinstance(bound, Decimal)
                for bound in (check.min_value, check.max_value)
                if bound is not None
            ):
                return _bound_type_diagnostic(
                    column,
                    actual,
                    "integer or float bounds for a non-decimal numeric column",
                )
        elif isinstance(check, MatchesRegex):
            if not isinstance(actual, StringType):
                expected = "string"
            else:
                regex_error = self._validate_regex(check)
                if regex_error is not None:
                    return regex_error
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
                or (isinstance(value, float) and isinstance(actual, (FloatType, DoubleType)))
                or (isinstance(value, Decimal) and isinstance(actual, DecimalType))
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

    def _validate_regex(self, check: MatchesRegex) -> Diagnostic | None:
        jvm = self._dataframe.sparkSession.sparkContext._jvm
        if jvm is None:
            raise RuntimeError("Spark JVM is unavailable for regex validation")
        try:
            jvm.java.util.regex.Pattern.compile(check.pattern)
        except IllegalArgumentException as error:
            message = str(error).splitlines()[0][:300]
            return Diagnostic(
                code="INVALID_REGEX",
                message=f"Pattern is not valid for Spark/JVM regex evaluation: {message}",
                column=check.column,
            )
        return None


def _column(name: str) -> Column:
    escaped = name.replace("`", "``")
    return F.col(f"`{escaped}`")


def _bound_type_diagnostic(column: str, actual: DataType, expected_bounds: str) -> Diagnostic:
    return Diagnostic(
        code="INCOMPATIBLE_BOUND_TYPE",
        message=f"Column {column!r} requires {expected_bounds}",
        column=column,
        details={
            "actual_type": actual.simpleString(),
            "expected_bounds": expected_bounds,
        },
    )


def _invalid_count(column: str, invalid: Column) -> Column:
    return F.sum(F.when(_column(column).isNotNull() & invalid, 1).otherwise(0))


def _range_invalid_expression(check: InRange, data_type: DataType) -> Column:
    value = _column(check.column)
    invalid = F.isnan(value) if isinstance(data_type, (FloatType, DoubleType)) else F.lit(False)
    if check.min_value is not None:
        below_minimum = value < check.min_value if check.inclusive else value <= check.min_value
        invalid = invalid | below_minimum
    if check.max_value is not None:
        above_maximum = value > check.max_value if check.inclusive else value >= check.max_value
        invalid = invalid | above_maximum
    return invalid


def _range_args(check: InRange) -> tuple[MetricArgument, ...]:
    return (check.min_value, check.max_value, check.inclusive)
