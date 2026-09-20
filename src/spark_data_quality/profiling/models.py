"""Serializable profiling result models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spark_data_quality.models.values import NumericValue


class TopValue(BaseModel):
    """One bounded categorical frequency."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    value: str
    count: int = Field(ge=0)


class NumericStatistics(BaseModel):
    """Statistics available for numeric columns."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    nan_count: int | None = Field(default=None, ge=0)
    minimum: NumericValue | None
    maximum: NumericValue | None
    mean: NumericValue | None
    standard_deviation: NumericValue | None


class StringStatistics(BaseModel):
    """Statistics available for string columns."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    minimum_length: int | None
    maximum_length: int | None
    top_values: tuple[TopValue, ...]


class ColumnProfile(BaseModel):
    """Profile for one selected column."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    column: str
    spark_type: str
    kind: Literal["numeric", "string"]
    null_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    numeric: NumericStatistics | None = None
    string: StringStatistics | None = None

    @model_validator(mode="after")
    def validate_statistics_kind(self) -> Self:
        if self.kind == "numeric" and (self.numeric is None or self.string is not None):
            raise ValueError("numeric profiles require only numeric statistics")
        if self.kind == "string" and (self.string is None or self.numeric is not None):
            raise ValueError("string profiles require only string statistics")
        return self


class ProfileExecutionMetadata(BaseModel):
    """Bounded execution facts for a profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    aggregate_actions: int = Field(ge=0)
    aggregate_metric_count: int = Field(ge=0)
    top_value_actions: int = Field(ge=0)
    top_k: int = Field(ge=1, le=100)
    scalar_aggregation_duration_ms: float = Field(ge=0.0)
    top_values_duration_ms: float = Field(ge=0.0)


class ProfileReport(BaseModel):
    """DataFrame profile detached from Spark."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="strings")

    row_count: int = Field(ge=0)
    columns: tuple[ColumnProfile, ...]
    metadata: ProfileExecutionMetadata
    total_duration_ms: float = Field(ge=0.0)
