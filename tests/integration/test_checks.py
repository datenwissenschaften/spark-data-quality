"""Built-in checks executed against a real local SparkSession."""

from collections.abc import Callable

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from spark_data_quality import (
    AllowedValues,
    Check,
    CheckStatus,
    ColumnExists,
    HasType,
    InRange,
    MatchesRegex,
    NotNull,
    QualitySuite,
    RowCount,
    Unique,
)


@pytest.mark.integration
def test_complete_suite_passes(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1, 18, "DE", "alice"), (2, 42, "ZA", "bob")],
        "player_id long, age long, country string, handle string",
    )
    suite = QualitySuite(
        "players",
        [
            ColumnExists("player_id"),
            HasType("player_id", LongType()),
            NotNull("player_id"),
            Unique("player_id"),
            InRange("age", min_value=13, max_value=120),
            AllowedValues("country", frozenset({"DE", "ZA", "GB"})),
            MatchesRegex("handle", r"^[a-z]+$"),
            RowCount(min_count=1, max_count=10),
        ],
    )

    report = suite.validate(dataframe)

    assert report.status is CheckStatus.PASS
    assert [result.status for result in report.results] == [CheckStatus.PASS] * 8
    assert report.metadata.aggregate_actions == 1
    assert report.metadata.schema_check_count == 2


@pytest.mark.integration
def test_data_violations_are_failures_with_diagnostics(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1, 10, "US", "Alice"), (1, 200, "DE", "ok"), (None, None, None, None)],
        "player_id long, age long, country string, handle string",
    )
    suite = QualitySuite(
        "players",
        [
            NotNull("player_id"),
            Unique("player_id"),
            InRange("age", min_value=13, max_value=120),
            AllowedValues("country", frozenset({"DE", "ZA"})),
            MatchesRegex("handle", r"^[a-z]+$"),
            RowCount(max_count=2),
        ],
    )

    report = suite.validate(dataframe)

    assert report.status is CheckStatus.FAIL
    assert [result.affected_rows for result in report.results[:5]] == [1, 1, 2, 1, 1]
    assert report.results[0].failure_fraction == pytest.approx(1 / 3)
    assert report.results[1].observed_value == {
        "non_null_count": 2,
        "distinct_count": 1,
        "duplicate_excess_count": 1,
    }
    assert report.results[-1].status is CheckStatus.FAIL
    assert report.results[-1].observed_value == 3


CheckFactory = Callable[[str], Check]


@pytest.mark.integration
@pytest.mark.parametrize(
    "factory",
    [
        lambda column: HasType(column, StringType()),
        NotNull,
        Unique,
        lambda column: InRange(column, min_value=0),
        lambda column: AllowedValues(column, frozenset({"x"})),
        lambda column: MatchesRegex(column, "x"),
    ],
)
def test_missing_column_is_error(
    spark: SparkSession,
    factory: CheckFactory,
) -> None:
    dataframe = spark.createDataFrame([(1,)], "present long")

    result = QualitySuite("missing", [factory("absent")]).validate(dataframe).results[0]

    assert result.status is CheckStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code == "MISSING_COLUMN"
    assert result.diagnostic.column == "absent"


@pytest.mark.integration
def test_column_exists_missing_is_fail(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], "present long")

    result = QualitySuite("exists", [ColumnExists("absent")]).validate(dataframe).results[0]

    assert result.status is CheckStatus.FAIL
    assert result.observed_value is False
    assert result.diagnostic is None


@pytest.mark.integration
def test_has_type_mismatch_is_fail_not_error(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], "id long")

    result = QualitySuite("types", [HasType("id", StringType())]).validate(dataframe).results[0]

    assert result.status is CheckStatus.FAIL
    assert result.observed_value == "bigint"
    assert result.diagnostic is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "check",
    [
        InRange("text", min_value=0),
        MatchesRegex("number", r"^1$"),
        AllowedValues("number", frozenset({"1"})),
    ],
)
def test_unexpected_spark_type_is_error(spark: SparkSession, check: Check) -> None:
    dataframe = spark.createDataFrame([("x", 1)], "text string, number long")

    result = QualitySuite("types", [check]).validate(dataframe).results[0]

    assert result.status is CheckStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code == "UNEXPECTED_SPARK_TYPE"


@pytest.mark.integration
def test_null_semantics_are_composable(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(None, None, None), (None, None, None)],
        "number long, category string, text string",
    )
    suite = QualitySuite(
        "nulls",
        [
            Unique("number"),
            InRange("number", min_value=0),
            AllowedValues("category", frozenset({"ok"})),
            MatchesRegex("text", r"^ok$"),
            NotNull("number"),
        ],
    )

    results = suite.validate(dataframe).results

    assert [result.status for result in results] == [
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.FAIL,
    ]
    assert [result.affected_rows for result in results] == [0, 0, 0, 0, 2]


@pytest.mark.integration
def test_empty_dataframe_semantics(spark: SparkSession) -> None:
    schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("text", StringType(), True),
        ]
    )
    dataframe = spark.createDataFrame([], schema)
    suite = QualitySuite(
        "empty",
        [
            ColumnExists("id"),
            HasType("id", LongType()),
            NotNull("id"),
            Unique("id"),
            InRange("id", min_value=0),
            AllowedValues("text", frozenset({"x"})),
            MatchesRegex("text", r"^x$"),
            RowCount(max_count=0),
            RowCount(min_count=1),
        ],
    )

    results = suite.validate(dataframe).results

    assert [result.status for result in results] == [CheckStatus.PASS] * 8 + [CheckStatus.FAIL]
    assert results[2].failure_fraction == 0.0


@pytest.mark.integration
def test_inclusive_and_exclusive_ranges(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,), (2,)], "value long")
    suite = QualitySuite(
        "ranges",
        [
            InRange("value", min_value=1, max_value=2),
            InRange("value", min_value=1, max_value=2, inclusive=False),
        ],
    )

    results = suite.validate(dataframe).results

    assert results[0].status is CheckStatus.PASS
    assert results[1].status is CheckStatus.FAIL
    assert results[1].affected_rows == 2


@pytest.mark.integration
def test_suite_error_takes_precedence_over_failure(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], "id long")
    suite = QualitySuite("precedence", [RowCount(min_count=2), NotNull("missing")])

    report = suite.validate(dataframe)

    assert [result.status for result in report.results] == [CheckStatus.FAIL, CheckStatus.ERROR]
    assert report.status is CheckStatus.ERROR


@pytest.mark.integration
def test_compatible_checks_share_one_aggregation_action(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1, "A"), (2, "B")], "id long, code string")
    suite = QualitySuite(
        "shared",
        [
            NotNull("id"),
            NotNull("id"),
            Unique("id"),
            InRange("id", min_value=0),
            AllowedValues("code", frozenset({"A", "B"})),
            RowCount(min_count=1),
        ],
    )

    report = suite.validate(dataframe)

    assert report.metadata.aggregate_actions == 1
    assert report.metadata.aggregate_metric_count == 6
    assert report.metadata.planned_groups == ("shared_aggregate",)


@pytest.mark.integration
def test_report_is_json_serializable_after_spark_values_are_collected(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame([(1,)], "id long")
    report = QualitySuite("json", [NotNull("id")]).validate(dataframe)

    payload = report.model_dump_json()

    assert '"suite_name":"json"' in payload
    assert '"status":"PASS"' in payload
    assert "SparkSession" not in payload


@pytest.mark.integration
def test_integer_allowed_values_require_integral_column(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))

    result = (
        QualitySuite("allowed", [AllowedValues("id", frozenset({1, 2}))])
        .validate(dataframe)
        .results[0]
    )

    assert result.status is CheckStatus.PASS
