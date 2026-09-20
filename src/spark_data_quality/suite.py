"""Public suite orchestration."""

from __future__ import annotations

from collections.abc import Sequence
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

    def __init__(self, name: str, checks: Sequence[Check]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("suite name must be non-empty")
        if not checks:
            raise ValueError("suite must contain at least one check")
        if any(not isinstance(check, Check) for check in checks):
            raise TypeError("suite checks must be Check instances")
        resolved_ids = [
            check.check_id or f"{check.check_type}:{index}" for index, check in enumerate(checks)
        ]
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("suite check identifiers must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "checks", tuple(checks))

    def validate(self, dataframe: DataFrame) -> SuiteReport:
        """Plan and evaluate this suite against a PySpark DataFrame."""

        started = perf_counter()
        planning_started = perf_counter()
        plan = AggregationPlanner(dataframe).plan(self.checks)
        planning_duration_ms = (perf_counter() - planning_started) * 1_000
        output = execute_plan(dataframe, plan)
        evaluation_started = perf_counter()
        results = tuple(evaluate_check(item, output.metrics) for item in plan.checks)
        result_evaluation_duration_ms = (perf_counter() - evaluation_started) * 1_000
        status = _suite_status(tuple(result.status for result in results))
        spark_context = dataframe.sparkSession.sparkContext
        metadata = ExecutionMetadata(
            spark_version=dataframe.sparkSession.version,
            spark_application_id=spark_context.applicationId,
            aggregate_actions=output.aggregate_actions,
            aggregate_metric_count=len(plan.metrics),
            schema_check_count=plan.schema_check_count,
            planned_groups=("shared_aggregate",) if plan.metrics else (),
            planning_duration_ms=planning_duration_ms,
            spark_execution_duration_ms=output.spark_execution_duration_ms,
            result_evaluation_duration_ms=result_evaluation_duration_ms,
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
