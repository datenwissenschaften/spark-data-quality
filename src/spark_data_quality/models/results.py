"""Serializable validation result models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckStatus(StrEnum):
    """Outcome of evaluating an expectation."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class Diagnostic(BaseModel):
    """A bounded, machine-readable explanation of an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    column: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    """Result of one check, detached from the Spark session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str
    check_type: str
    description: str
    status: CheckStatus
    observed_value: Any | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    affected_rows: int | None = None
    total_rows: int | None = None
    failure_fraction: float | None = None
    execution_duration_ms: float
    diagnostic: Diagnostic | None = None


class ExecutionMetadata(BaseModel):
    """Facts about how a suite was planned and executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spark_version: str
    spark_application_id: str
    aggregate_actions: int
    aggregate_metric_count: int
    schema_check_count: int
    planned_groups: tuple[str, ...]


class SuiteReport(BaseModel):
    """Complete result of a suite validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_name: str
    status: CheckStatus
    results: tuple[CheckResult, ...]
    metadata: ExecutionMetadata
    total_duration_ms: float
