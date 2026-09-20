"""Small, bounded Spark-native DataFrame profiler."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

from spark_data_quality.exceptions import InvalidProfileError
from spark_data_quality.profiling.models import (
    ColumnProfile,
    NumericStatistics,
    ProfileExecutionMetadata,
    ProfileReport,
    StringStatistics,
    TopValue,
)


def profile(
    dataframe: DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    top_k: int = 10,
) -> ProfileReport:
    """Profile selected numeric and string columns with bounded collection."""

    started = perf_counter()
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")

    selected = tuple(columns) if columns is not None else tuple(dataframe.columns)
    if len(set(selected)) != len(selected):
        raise InvalidProfileError("profile columns must be unique")

    missing = [column for column in selected if column not in dataframe.columns]
    if missing:
        raise InvalidProfileError(f"Profile column {missing[0]!r} does not exist")

    unsupported = [
        column
        for column in selected
        if not isinstance(dataframe.schema[column].dataType, (NumericType, StringType))
    ]
    if unsupported:
        data_type = dataframe.schema[unsupported[0]].dataType.simpleString()
        raise InvalidProfileError(
            f"Profile column {unsupported[0]!r} has unsupported Spark type {data_type!r}"
        )

    expressions: list[Column] = [F.count(F.lit(1)).alias("row_count")]
    aliases: dict[str, dict[str, str]] = {}
    for index, column in enumerate(selected):
        prefix = f"c{index:04d}"
        aliases[column] = {
            "null": f"{prefix}_null",
            "distinct": f"{prefix}_distinct",
        }
        value = _column(column)
        expressions.extend(
            [
                F.sum(F.when(value.isNull(), 1).otherwise(0)).alias(aliases[column]["null"]),
                F.countDistinct(value).alias(aliases[column]["distinct"]),
            ]
        )
        if isinstance(dataframe.schema[column].dataType, NumericType):
            aliases[column].update(
                {
                    "min": f"{prefix}_min",
                    "max": f"{prefix}_max",
                    "mean": f"{prefix}_mean",
                    "stddev": f"{prefix}_stddev",
                }
            )
            expressions.extend(
                [
                    F.min(value).alias(aliases[column]["min"]),
                    F.max(value).alias(aliases[column]["max"]),
                    F.mean(value).alias(aliases[column]["mean"]),
                    F.stddev_samp(value).alias(aliases[column]["stddev"]),
                ]
            )
        else:
            aliases[column].update(
                {"min_length": f"{prefix}_min_length", "max_length": f"{prefix}_max_length"}
            )
            expressions.extend(
                [
                    F.min(F.length(value)).alias(aliases[column]["min_length"]),
                    F.max(F.length(value)).alias(aliases[column]["max_length"]),
                ]
            )

    scalar_row = dataframe.agg(*expressions).first()
    if scalar_row is None:
        raise RuntimeError("Spark aggregate unexpectedly returned no result row")
    metrics = scalar_row.asDict(recursive=True)
    profiles: list[ColumnProfile] = []
    top_value_actions = 0

    for column in selected:
        spark_type = dataframe.schema[column].dataType
        common: dict[str, Any] = {
            "column": column,
            "spark_type": spark_type.simpleString(),
            "null_count": int(metrics[aliases[column]["null"]] or 0),
            "distinct_count": int(metrics[aliases[column]["distinct"]] or 0),
        }
        if isinstance(spark_type, NumericType):
            profiles.append(
                ColumnProfile(
                    **common,
                    kind="numeric",
                    numeric=NumericStatistics(
                        minimum=metrics[aliases[column]["min"]],
                        maximum=metrics[aliases[column]["max"]],
                        mean=metrics[aliases[column]["mean"]],
                        standard_deviation=metrics[aliases[column]["stddev"]],
                    ),
                )
            )
        else:
            top_rows = (
                dataframe.where(_column(column).isNotNull())
                .groupBy(_column(column).alias("value"))
                .count()
                .orderBy(F.desc("count"), F.asc("value"))
                .limit(top_k)
                .collect()
            )
            top_value_actions += 1
            profiles.append(
                ColumnProfile(
                    **common,
                    kind="string",
                    string=StringStatistics(
                        minimum_length=metrics[aliases[column]["min_length"]],
                        maximum_length=metrics[aliases[column]["max_length"]],
                        top_values=tuple(
                            TopValue(value=row["value"], count=int(row["count"]))
                            for row in top_rows
                        ),
                    ),
                )
            )

    return ProfileReport(
        row_count=int(metrics["row_count"]),
        columns=tuple(profiles),
        metadata=ProfileExecutionMetadata(
            aggregate_actions=1,
            aggregate_metric_count=len(expressions),
            top_value_actions=top_value_actions,
            top_k=top_k,
        ),
        total_duration_ms=(perf_counter() - started) * 1_000,
    )


def _column(name: str) -> Column:
    escaped = name.replace("`", "``")
    return F.col(f"`{escaped}`")
