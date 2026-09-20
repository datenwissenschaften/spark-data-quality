"""Profiler integration tests."""

import json
import math
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DecimalType, DoubleType, FloatType, StructField, StructType

from spark_data_quality import profile
from spark_data_quality.exceptions import InvalidProfileError
from spark_data_quality.models import NonFiniteFloat


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


@pytest.mark.integration
def test_profile_rejects_duplicate_columns(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], "id long")

    with pytest.raises(InvalidProfileError, match="must be unique"):
        profile(dataframe, ["id", "id"])


@pytest.mark.integration
@pytest.mark.parametrize("spark_type", [FloatType(), DoubleType()])
def test_numeric_profile_preserves_special_float_semantics(
    spark: SparkSession,
    spark_type: FloatType | DoubleType,
) -> None:
    dataframe = spark.createDataFrame(
        [(1.0,), (math.nan,), (math.nan,), (math.inf,), (-math.inf,), (None,)],
        StructType([StructField("value", spark_type)]),
    )

    report = profile(dataframe, ["value"])

    column = report.columns[0]
    assert column.null_count == 1
    assert column.distinct_count == 4
    assert column.numeric is not None
    assert column.numeric.nan_count == 2
    assert column.numeric.minimum is NonFiniteFloat.NEGATIVE_INFINITY
    assert column.numeric.maximum is NonFiniteFloat.NAN
    assert column.numeric.mean is NonFiniteFloat.NAN
    assert column.numeric.standard_deviation is NonFiniteFloat.NAN
    payload = report.model_dump_json()
    assert json.loads(payload)["columns"][0]["numeric"]["maximum"] == "NaN"
    assert ":NaN" not in payload


@pytest.mark.integration
def test_numeric_profile_reports_positive_infinity_without_nan(spark: SparkSession) -> None:
    """Exercise the positive-infinity branch on its own, distinct from the NaN case.

    When NaN is also present it sorts above +Infinity and becomes the maximum
    instead, so this path needs data that reaches +Infinity without any NaN.
    """

    dataframe = spark.createDataFrame(
        [(1.0,), (math.inf,)],
        StructType([StructField("value", DoubleType())]),
    )

    report = profile(dataframe, ["value"])

    numeric = report.columns[0].numeric
    assert numeric is not None
    assert numeric.nan_count == 0
    assert numeric.maximum is NonFiniteFloat.POSITIVE_INFINITY
    assert numeric.mean is NonFiniteFloat.POSITIVE_INFINITY


@pytest.mark.integration
def test_decimal_profile_retains_exact_decimal_values(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [
            (Decimal("0.123456789012345678"),),
            (Decimal("0.123456789012345679"),),
        ],
        StructType([StructField("value", DecimalType(30, 18), False)]),
    )

    report = profile(dataframe, ["value"])

    numeric = report.columns[0].numeric
    assert numeric is not None
    assert numeric.nan_count is None
    assert numeric.minimum == Decimal("0.123456789012345678")
    assert numeric.maximum == Decimal("0.123456789012345679")
    assert numeric.mean == Decimal("0.1234567890123456785000")
    # Spark's stddev_samp always returns DoubleType, even for a DecimalType
    # input column, so standard_deviation is a float while the other decimal
    # statistics above remain exact Decimal values. This is documented Spark
    # aggregate behavior, not a library conversion.
    assert isinstance(numeric.standard_deviation, float)
    payload = json.loads(report.model_dump_json())
    assert payload["columns"][0]["numeric"]["minimum"] == "0.123456789012345678"


@pytest.mark.integration
def test_top_values_are_bounded_for_high_cardinality_and_include_empty_string(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame(
        [("",)] + [(f"value-{index:04d}",) for index in range(250)],
        "value string",
    )

    report = profile(dataframe, ["value"], top_k=3)

    string = report.columns[0].string
    assert string is not None
    assert len(string.top_values) == 3
    assert [item.value for item in string.top_values] == ["", "value-0000", "value-0001"]
    assert string.minimum_length == 0


@pytest.mark.integration
def test_profile_with_no_selected_columns_is_serializable(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,), (2,)], "id long")

    report = profile(dataframe, [])

    assert report.row_count == 2
    assert report.columns == ()
    assert json.loads(report.model_dump_json())["columns"] == []
