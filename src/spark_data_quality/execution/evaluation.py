"""Interpret scalar Spark metrics as structured check results."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

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


def evaluate_check(
    prepared: PreparedCheck,
    metrics: Mapping[str, Any],
    aggregation_duration_ms: float,
) -> CheckResult:
    """Build a public result from one prepared check and scalar metrics."""

    started = perf_counter()
    check = prepared.check
    duration = prepared.planning_duration_ms

    if prepared.diagnostic is not None:
        return CheckResult(
            check_id=prepared.check_id,
            check_type=check.check_type,
            description=check.description,
            status=CheckStatus.ERROR,
            expected=check.expected(),
            execution_duration_ms=duration + _elapsed_ms(started),
            diagnostic=prepared.diagnostic,
        )

    if isinstance(check, ColumnExists):
        observed_exists = bool(prepared.schema_observation)
        return _result(
            prepared,
            CheckStatus.PASS if observed_exists else CheckStatus.FAIL,
            observed=observed_exists,
            duration_ms=duration + _elapsed_ms(started),
        )

    if isinstance(check, HasType):
        schema_observed = str(prepared.schema_observation)
        status = (
            CheckStatus.PASS
            if schema_observed == check.expected_type.simpleString()
            else CheckStatus.FAIL
        )
        return _result(
            prepared,
            status,
            observed=schema_observed,
            duration_ms=duration + _elapsed_ms(started),
        )

    duration += aggregation_duration_ms
    total = int(metrics[prepared.metric_aliases["total"]])

    if isinstance(check, RowCount):
        passes_min = check.min_count is None or total >= check.min_count
        passes_max = check.max_count is None or total <= check.max_count
        return _result(
            prepared,
            CheckStatus.PASS if passes_min and passes_max else CheckStatus.FAIL,
            observed=total,
            total_rows=total,
            duration_ms=duration + _elapsed_ms(started),
        )

    if isinstance(check, Unique):
        non_null = int(metrics[prepared.metric_aliases["non_null"]])
        distinct = int(metrics[prepared.metric_aliases["distinct"]])
        affected = non_null - distinct
        data_observed: Any = {
            "non_null_count": non_null,
            "distinct_count": distinct,
            "duplicate_excess_count": affected,
        }
    elif isinstance(check, (NotNull, InRange, AllowedValues, MatchesRegex)):
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
        failure_fraction=affected / total if total else 0.0,
        duration_ms=duration + _elapsed_ms(started),
    )


def _result(
    prepared: PreparedCheck,
    status: CheckStatus,
    *,
    observed: Any,
    duration_ms: float,
    affected_rows: int | None = None,
    total_rows: int | None = None,
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
        failure_fraction=failure_fraction,
        execution_duration_ms=duration_ms,
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000
