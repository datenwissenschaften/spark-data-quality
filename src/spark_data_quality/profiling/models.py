"""Serializable profiling result models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TopValue(BaseModel):
    """One bounded categorical frequency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Any
    count: int


class NumericStatistics(BaseModel):
    """Statistics available for numeric columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: int | float | None
    maximum: int | float | None
    mean: float | None
    standard_deviation: float | None


class StringStatistics(BaseModel):
    """Statistics available for string columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_length: int | None
    maximum_length: int | None
    top_values: tuple[TopValue, ...]


class ColumnProfile(BaseModel):
    """Profile for one selected column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    spark_type: str
    kind: Literal["numeric", "string"]
    null_count: int
    distinct_count: int
    numeric: NumericStatistics | None = None
    string: StringStatistics | None = None


class ProfileExecutionMetadata(BaseModel):
    """Bounded execution facts for a profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregate_actions: int
    aggregate_metric_count: int
    top_value_actions: int
    top_k: int


class ProfileReport(BaseModel):
    """DataFrame profile detached from Spark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int
    columns: tuple[ColumnProfile, ...]
    metadata: ProfileExecutionMetadata
    total_duration_ms: float
