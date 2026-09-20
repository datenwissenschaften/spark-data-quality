"""Result-model consistency and standards-compatible serialization tests."""

import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from spark_data_quality.models import (
    CheckResult,
    CheckStatus,
    Diagnostic,
    ExecutionMetadata,
    SuiteReport,
)


def _metadata() -> ExecutionMetadata:
    return ExecutionMetadata(
        spark_version="4.1.3",
        spark_application_id="local-test",
        aggregate_actions=0,
        aggregate_metric_count=0,
        schema_check_count=1,
        planned_groups=(),
        planning_duration_ms=0.1,
        spark_execution_duration_ms=0.0,
        result_evaluation_duration_ms=0.1,
    )


def test_report_serializes_domain_values_as_standard_json() -> None:
    result = CheckResult(
        check_id="serialization",
        check_type="test",
        description="serialization test",
        status=CheckStatus.PASS,
        observed_value={
            "decimal": Decimal("1234567890.123456789"),
            "date": date(2026, 9, 21),
            "datetime": datetime(2026, 9, 21, 12, 30, tzinfo=UTC),
            "nan": math.nan,
            "positive_infinity": math.inf,
            "negative_infinity": -math.inf,
            "none": None,
        },
        expected={"status": CheckStatus.PASS.value},
    )
    report = SuiteReport(
        suite_name="serialization",
        status=CheckStatus.PASS,
        results=(result,),
        metadata=_metadata(),
        total_duration_ms=0.3,
    )

    payload = json.loads(report.model_dump_json())

    observed = payload["results"][0]["observed_value"]
    assert observed == {
        "decimal": "1234567890.123456789",
        "date": "2026-09-21",
        "datetime": "2026-09-21T12:30:00Z",
        "nan": "NaN",
        "positive_infinity": "Infinity",
        "negative_infinity": "-Infinity",
        "none": None,
    }
    assert payload["results"][0]["status"] == "PASS"


def test_error_result_requires_diagnostic() -> None:
    with pytest.raises(ValidationError, match="require a diagnostic"):
        CheckResult(
            check_id="error",
            check_type="test",
            description="error",
            status=CheckStatus.ERROR,
        )

    error = CheckResult(
        check_id="error",
        check_type="test",
        description="error",
        status=CheckStatus.ERROR,
        diagnostic=Diagnostic(code="TEST", message="test"),
    )
    assert error.diagnostic is not None


def test_diagnostic_is_rejected_outside_error_status() -> None:
    with pytest.raises(ValidationError, match="only ERROR results may contain"):
        CheckResult(
            check_id="mismatch",
            check_type="test",
            description="mismatch",
            status=CheckStatus.PASS,
            diagnostic=Diagnostic(code="TEST", message="test"),
        )


def test_evaluated_rows_cannot_exceed_total_rows() -> None:
    with pytest.raises(ValidationError, match="evaluated_rows must not exceed total_rows"):
        CheckResult(
            check_id="rows",
            check_type="test",
            description="rows",
            status=CheckStatus.PASS,
            total_rows=1,
            evaluated_rows=2,
        )


def test_affected_rows_cannot_exceed_evaluated_rows() -> None:
    with pytest.raises(ValidationError, match="affected_rows must not exceed evaluated_rows"):
        CheckResult(
            check_id="rows",
            check_type="test",
            description="rows",
            status=CheckStatus.FAIL,
            evaluated_rows=1,
            affected_rows=2,
        )


def test_failure_fraction_must_use_positive_evaluated_rows() -> None:
    with pytest.raises(ValidationError, match="positive evaluated_rows"):
        CheckResult(
            check_id="fraction",
            check_type="test",
            description="fraction",
            status=CheckStatus.PASS,
            affected_rows=0,
            total_rows=0,
            evaluated_rows=0,
            failure_fraction=0.0,
        )

    with pytest.raises(ValidationError, match="must equal"):
        CheckResult(
            check_id="fraction",
            check_type="test",
            description="fraction",
            status=CheckStatus.FAIL,
            affected_rows=1,
            total_rows=4,
            evaluated_rows=2,
            failure_fraction=0.25,
        )
