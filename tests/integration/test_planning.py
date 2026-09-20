"""Aggregation planning invariants tested with real Spark expressions."""

import re

import pytest
from pyspark.sql import SparkSession

from spark_data_quality import AllowedValues, InRange, NotNull, RowCount, Unique
from spark_data_quality.execution.planning import AggregationPlanner


@pytest.mark.integration
def test_metric_aliases_are_deterministic_unique_and_safe(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1, "A")], "id long, code string")
    checks = (
        NotNull("id"),
        NotNull("id"),
        Unique("id"),
        InRange("id", min_value=0, max_value=10),
        AllowedValues("code", {"A", "B"}),
        RowCount(min_count=1),
    )

    first = AggregationPlanner(dataframe).plan(checks)
    second = AggregationPlanner(dataframe).plan(checks)
    first_metrics = [(metric.key, metric.alias) for metric in first.metrics]
    second_metrics = [(metric.key, metric.alias) for metric in second.metrics]
    aliases = [metric.alias for metric in first.metrics]

    assert first_metrics == second_metrics
    assert len(aliases) == len(set(aliases))
    assert all(re.fullmatch(r"metric_[0-9]{4,}", alias) for alias in aliases)
    assert first.checks[0].metric_aliases == first.checks[1].metric_aliases


@pytest.mark.integration
def test_shared_aggregation_reads_the_source_exactly_once(spark: SparkSession) -> None:
    """Verify the "one shared scan" claim mechanically instead of only in prose.

    Counting Catalyst leaf nodes on the pre-AQE physical plan is stable across
    Spark versions, unlike asserting on exact operator names or counts.
    """

    dataframe = spark.createDataFrame([(1, "A"), (2, "B"), (3, None)], "id long, code string")
    checks = (
        NotNull("id"),
        Unique("id"),
        InRange("id", min_value=0),
        AllowedValues("code", {"A", "B"}),
        RowCount(min_count=1),
    )

    plan = AggregationPlanner(dataframe).plan(checks)
    aggregated = dataframe.agg(*(metric.expression for metric in plan.metrics))
    physical_plan = aggregated._jdf.queryExecution().sparkPlan()

    assert physical_plan.collectLeaves().length() == 1
