"""Public suite orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from pyspark.sql import DataFrame

from spark_data_quality.checks import Check
from spark_data_quality.execution.evaluation import evaluate_check
from spark_data_quality.execution.planning import AggregationPlanner
from spark_data_quality.execution.runner import execute_plan
from spark_data_quality.models import CheckStatus, ExecutionMetadata, SuiteReport


@dataclass(frozen=True, slots=True)
class QualitySuite:
    """An immutable, reusable collection of ordered expectations."""

    name: str
    checks: tuple[Check, ...]

    def __init__(self, name: str, checks: list[Check] | tuple[Check, ...]) -> None:
        if not name or not name.strip():
            raise ValueError("suite name must be non-empty")
        if not checks:
            raise ValueError("suite must contain at least one check")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "checks", tuple(checks))

    def validate(self, dataframe: DataFrame) -> SuiteReport:
        """Plan and evaluate this suite against a PySpark DataFrame."""

        started = perf_counter()
        plan = AggregationPlanner(dataframe).plan(self.checks)
        output = execute_plan(dataframe, plan)
        results = tuple(
            evaluate_check(item, output.metrics, output.aggregation_duration_ms)
            for item in plan.checks
        )
        status = _suite_status(tuple(result.status for result in results))
        spark_context = dataframe.sparkSession.sparkContext
        metadata = ExecutionMetadata(
            spark_version=dataframe.sparkSession.version,
            spark_application_id=spark_context.applicationId,
            aggregate_actions=output.aggregate_actions,
            aggregate_metric_count=len(plan.metrics),
            schema_check_count=plan.schema_check_count,
            planned_groups=("shared_aggregate",) if plan.metrics else (),
        )
        return SuiteReport(
            suite_name=self.name,
            status=status,
            results=results,
            metadata=metadata,
            total_duration_ms=(perf_counter() - started) * 1_000,
        )


def _suite_status(statuses: tuple[CheckStatus, ...]) -> CheckStatus:
    if CheckStatus.ERROR in statuses:
        return CheckStatus.ERROR
    if CheckStatus.FAIL in statuses:
        return CheckStatus.FAIL
    return CheckStatus.PASS
