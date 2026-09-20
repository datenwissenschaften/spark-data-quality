"""Profiler integration tests."""

import pytest
from pyspark.sql import SparkSession

from spark_data_quality import profile
from spark_data_quality.exceptions import InvalidProfileError


@pytest.mark.integration
def test_profiles_numeric_and_string_columns(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1, "DE"), (2, "ZA"), (3, "DE"), (None, None)],
        "score long, country string",
    )

    report = profile(dataframe, ["score", "country"], top_k=1)

    assert report.row_count == 4
    numeric = report.columns[0]
    assert numeric.null_count == 1
    assert numeric.distinct_count == 3
    assert numeric.numeric is not None
    assert numeric.numeric.minimum == 1
    assert numeric.numeric.maximum == 3
    assert numeric.numeric.mean == 2.0
    categorical = report.columns[1]
    assert categorical.string is not None
    assert categorical.string.minimum_length == 2
    assert categorical.string.maximum_length == 2
    assert [(item.value, item.count) for item in categorical.string.top_values] == [("DE", 2)]
    assert report.metadata.aggregate_actions == 1
    assert report.metadata.top_value_actions == 1
    assert len(report.model_dump_json()) > 0


@pytest.mark.integration
def test_profile_top_values_have_deterministic_tie_order(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([("B",), ("A",), ("C",)], "value string")

    report = profile(dataframe, ["value"], top_k=2)

    string = report.columns[0].string
    assert string is not None
    assert [item.value for item in string.top_values] == ["A", "B"]


@pytest.mark.integration
def test_empty_profile_has_null_statistics(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([], "number long, text string")

    report = profile(dataframe)

    assert report.row_count == 0
    assert report.columns[0].numeric is not None
    assert report.columns[0].numeric.mean is None
    assert report.columns[1].string is not None
    assert report.columns[1].string.top_values == ()


@pytest.mark.integration
def test_profile_rejects_missing_and_unsupported_columns(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(True,)], "flag boolean")

    with pytest.raises(InvalidProfileError, match="does not exist"):
        profile(dataframe, ["missing"])
    with pytest.raises(InvalidProfileError, match="unsupported"):
        profile(dataframe, ["flag"])
    with pytest.raises(ValueError, match="between 1 and 100"):
        profile(dataframe, [], top_k=0)
