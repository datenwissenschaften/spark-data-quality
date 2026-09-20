"""Interpret scalar Spark metrics as structured check results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyspark.sql.types import DataType

from spark_data_quality.checks import (
    AllowedValues,
    ColumnExists,
    HasType,
    InRange,
    MatchesRegex,
    NotNull,
    RowCount,
    Unique,
)
from spark_data_quality.execution.planning import PreparedCheck
from spark_data_quality.models import CheckResult, CheckStatus
from spark_data_quality.models.values import JsonValue


def evaluate_check(
    prepared: PreparedCheck,
    metrics: Mapping[str, Any],
) -> CheckResult:
    """Build a public result from one prepared check and scalar metrics."""

    check = prepared.check

    if prepared.diagnostic is not None:
        return CheckResult(
            check_id=prepared.check_id,
            check_type=check.check_type,
            description=check.description,
            status=CheckStatus.ERROR,
            expected=check.expected(),
            diagnostic=prepared.diagnostic,
        )

    if isinstance(check, ColumnExists):
        observed_exists = bool(prepared.schema_observation)
        return _result(
            prepared,
            CheckStatus.PASS if observed_exists else CheckStatus.FAIL,
            observed=observed_exists,
        )

    if isinstance(check, HasType):
        actual_type = prepared.schema_observation
        if not isinstance(actual_type, DataType):
            raise TypeError("HasType requires a planned Spark DataType observation")
        status = CheckStatus.PASS if actual_type == check.expected_type else CheckStatus.FAIL
        return _result(
            prepared,
            status,
            observed=actual_type.simpleString(),
        )

    total = int(metrics[prepared.metric_aliases["total"]])

    if isinstance(check, RowCount):
        passes_min = check.min_count is None or total >= check.min_count
        passes_max = check.max_count is None or total <= check.max_count
        return _result(
            prepared,
            CheckStatus.PASS if passes_min and passes_max else CheckStatus.FAIL,
            observed=total,
            total_rows=total,
        )

    if isinstance(check, Unique):
        evaluated = int(metrics[prepared.metric_aliases["evaluated"]])
        distinct = int(metrics[prepared.metric_aliases["distinct"]])
        affected = evaluated - distinct
        data_observed: JsonValue = {
            "non_null_count": evaluated,
            "distinct_count": distinct,
            "duplicate_excess_count": affected,
        }
    elif isinstance(check, NotNull):
        evaluated = total
        affected = int(metrics[prepared.metric_aliases["invalid"]] or 0)
        data_observed = affected
    elif isinstance(check, (InRange, AllowedValues, MatchesRegex)):
        evaluated = int(metrics[prepared.metric_aliases["evaluated"]])
        affected = int(metrics[prepared.metric_aliases["invalid"]] or 0)
        data_observed = affected
    else:
        raise TypeError(f"Unsupported check type: {type(check).__name__}")

    return _result(
        prepared,
        CheckStatus.PASS if affected == 0 else CheckStatus.FAIL,
        observed=data_observed,
        affected_rows=affected,
        total_rows=total,
        evaluated_rows=evaluated,
        failure_fraction=affected / evaluated if evaluated else None,
    )


def _result(
    prepared: PreparedCheck,
    status: CheckStatus,
    *,
    observed: JsonValue,
    affected_rows: int | None = None,
    total_rows: int | None = None,
    evaluated_rows: int | None = None,
    failure_fraction: float | None = None,
) -> CheckResult:
    check = prepared.check
    return CheckResult(
        check_id=prepared.check_id,
        check_type=check.check_type,
        description=check.description,
        status=status,
        observed_value=observed,
        expected=check.expected(),
        affected_rows=affected_rows,
        total_rows=total_rows,
        evaluated_rows=evaluated_rows,
        failure_fraction=failure_fraction,
    )
