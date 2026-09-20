"""Serializable validation result models."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spark_data_quality.models.values import JsonValue


class CheckStatus(StrEnum):
    """Outcome of evaluating an expectation."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Diagnostic(BaseModel):
    """A bounded, machine-readable explanation of an error."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    code: str
    message: str
    column: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class CheckResult(BaseModel):
    """Result of one check, detached from the Spark session."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    check_id: str
    check_type: str
    description: str
    status: CheckStatus
    observed_value: JsonValue = None
    expected: dict[str, JsonValue] = Field(default_factory=dict)
    affected_rows: int | None = Field(default=None, ge=0)
    total_rows: int | None = Field(default=None, ge=0)
    evaluated_rows: int | None = Field(default=None, ge=0)
    failure_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic: Diagnostic | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> Self:
        if self.status is CheckStatus.ERROR and self.diagnostic is None:
            raise ValueError("ERROR results require a diagnostic")
        if self.status is not CheckStatus.ERROR and self.diagnostic is not None:
            raise ValueError("only ERROR results may contain a diagnostic")
        if (
            self.evaluated_rows is not None
            and self.total_rows is not None
            and self.evaluated_rows > self.total_rows
        ):
            raise ValueError("evaluated_rows must not exceed total_rows")
        if (
            self.affected_rows is not None
            and self.evaluated_rows is not None
            and self.affected_rows > self.evaluated_rows
        ):
            raise ValueError("affected_rows must not exceed evaluated_rows")
        if self.failure_fraction is not None:
            if (
                self.affected_rows is None
                or self.evaluated_rows is None
                or self.evaluated_rows == 0
            ):
                raise ValueError(
                    "failure_fraction requires affected_rows and positive evaluated_rows"
                )
            expected_fraction = self.affected_rows / self.evaluated_rows
            if abs(self.failure_fraction - expected_fraction) > 1e-12:
                raise ValueError("failure_fraction must equal affected_rows / evaluated_rows")
        return self


class ExecutionMetadata(BaseModel):
    """Facts about how a suite was planned and executed."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    spark_version: str
    spark_application_id: str
    aggregate_actions: int = Field(ge=0)
    aggregate_metric_count: int = Field(ge=0)
    schema_check_count: int = Field(ge=0)
    planned_groups: tuple[str, ...]
    planning_duration_ms: float = Field(ge=0.0)
    spark_execution_duration_ms: float = Field(ge=0.0)
    result_evaluation_duration_ms: float = Field(ge=0.0)


class SuiteReport(BaseModel):
    """Complete result of a suite validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    suite_name: str
    status: CheckStatus
    results: tuple[CheckResult, ...]
    metadata: ExecutionMetadata
    total_duration_ms: float = Field(ge=0.0)
