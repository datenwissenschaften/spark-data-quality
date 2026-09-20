"""Execute aggregation plans against Spark."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pyspark.sql import DataFrame

from spark_data_quality.execution.planning import AggregationPlan


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    """Bounded scalar output from executing a plan."""

    metrics: dict[str, Any]
    aggregate_actions: int
    spark_execution_duration_ms: float


def execute_plan(dataframe: DataFrame, plan: AggregationPlan) -> ExecutionOutput:
    """Run the plan, collecting at most one aggregate row to the driver."""

    if not plan.metrics:
        return ExecutionOutput(metrics={}, aggregate_actions=0, spark_execution_duration_ms=0.0)

    started = perf_counter()
    row = dataframe.agg(*(metric.expression for metric in plan.metrics)).first()
    if row is None:
        raise RuntimeError("Spark aggregate unexpectedly returned no result row")
    elapsed_ms = (perf_counter() - started) * 1_000
    return ExecutionOutput(
        metrics=row.asDict(recursive=True),
        aggregate_actions=1,
        spark_execution_duration_ms=elapsed_ms,
    )
