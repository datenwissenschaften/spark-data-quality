"""Built-in checks executed against a real local SparkSession."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
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
from spark_data_quality.models.values import JsonValue


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
    assert report.results[0].evaluated_rows == 3
    assert report.results[2].evaluated_rows == 2
    assert report.results[2].failure_fraction == 1.0
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
    assert results[2].failure_fraction is None
    assert results[2].evaluated_rows == 0


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
def test_compatible_checks_share_one_aggregation_action(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataframe = spark.createDataFrame([(1, "A"), (2, "B")], "id long, code string")
    dataframe_type = type(dataframe)
    original_agg = dataframe_type.agg
    aggregate_calls: list[int] = []

    def tracked_agg(frame: DataFrame, *expressions: object) -> DataFrame:
        aggregate_calls.append(len(expressions))
        return original_agg(frame, *expressions)  # type: ignore[arg-type]

    monkeypatch.setattr(dataframe_type, "agg", tracked_agg)
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

    assert aggregate_calls == [7]
    assert report.metadata.aggregate_actions == 1
    assert report.metadata.aggregate_metric_count == 7
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


@pytest.mark.integration
def test_has_type_uses_exact_spark_datatype_equality(spark: SparkSession) -> None:
    fields = [
        StructField("string", StringType()),
        StructField("boolean", BooleanType()),
        StructField("byte", ByteType()),
        StructField("short", ShortType()),
        StructField("integer", IntegerType()),
        StructField("long", LongType()),
        StructField("float", FloatType()),
        StructField("double", DoubleType()),
        StructField("decimal", DecimalType(20, 4)),
        StructField("date", DateType()),
        StructField("timestamp", TimestampType()),
    ]
    dataframe = spark.createDataFrame([], StructType(fields))
    checks = [HasType(field.name, field.dataType) for field in fields]
    checks.append(HasType("decimal", DecimalType(20, 5)))

    results = QualitySuite("exact-types", checks).validate(dataframe).results

    assert [result.status for result in results[:-1]] == [CheckStatus.PASS] * len(fields)
    assert results[-1].status is CheckStatus.FAIL
    assert results[-1].observed_value == "decimal(20,4)"


@pytest.mark.integration
def test_nan_null_and_infinity_semantics(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1.0,), (None,), (math.nan,), (math.nan,), (math.inf,), (-math.inf,)],
        StructType([StructField("value", DoubleType())]),
    )
    suite = QualitySuite(
        "special-floats",
        [
            NotNull("value"),
            Unique("value"),
            InRange("value", min_value=-10.0, max_value=10.0),
            InRange("value", min_value=-10.0),
            InRange("value", max_value=10.0),
        ],
    )

    results = suite.validate(dataframe).results

    assert results[0].affected_rows == 1  # NaN and infinities are not SQL NULL.
    assert results[0].evaluated_rows == 6
    assert results[1].observed_value == {
        "non_null_count": 5,
        "distinct_count": 4,
        "duplicate_excess_count": 1,
    }
    assert results[1].failure_fraction == pytest.approx(1 / 5)
    assert results[2].affected_rows == 4  # two NaNs and both infinities
    assert results[3].affected_rows == 3  # two NaNs and negative infinity
    assert results[4].affected_rows == 3  # two NaNs and positive infinity


@pytest.mark.integration
def test_float_nan_is_always_out_of_range(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1.0,), (math.nan,)],
        StructType([StructField("value", FloatType())]),
    )

    result = (
        QualitySuite("float-nan", [InRange("value", min_value=0.0)]).validate(dataframe).results[0]
    )

    assert result.status is CheckStatus.FAIL
    assert result.affected_rows == 1


@pytest.mark.integration
def test_decimal_range_uses_decimal_bounds_without_float_conversion(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame(
        [
            (Decimal("0.123456789012345678"),),
            (Decimal("0.123456789012345679"),),
            (None,),
        ],
        StructType([StructField("value", DecimalType(30, 18))]),
    )
    exact = InRange(
        "value",
        min_value=Decimal("0.123456789012345678"),
        max_value=Decimal("0.123456789012345678"),
    )

    result = QualitySuite("decimal", [exact]).validate(dataframe).results[0]
    incompatible = (
        QualitySuite("decimal-float", [InRange("value", min_value=0.1)])
        .validate(dataframe)
        .results[0]
    )

    assert result.status is CheckStatus.FAIL
    assert result.affected_rows == 1
    assert result.evaluated_rows == 2
    assert incompatible.status is CheckStatus.ERROR
    assert incompatible.diagnostic is not None
    assert incompatible.diagnostic.code == "INCOMPATIBLE_BOUND_TYPE"


@pytest.mark.integration
def test_decimal_bound_on_non_decimal_column_is_error(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,), (2,)], "value long")

    result = (
        QualitySuite("decimal-bound", [InRange("value", min_value=Decimal("1"))])
        .validate(dataframe)
        .results[0]
    )

    assert result.status is CheckStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code == "INCOMPATIBLE_BOUND_TYPE"


@dataclass(frozen=True, slots=True)
class _UnsupportedCheck(Check):
    """A minimal, non-built-in Check used to test planner extensibility limits."""

    column: str
    check_type: ClassVar[str] = "unsupported"

    @property
    def description(self) -> str:
        return f"Column {self.column!r} satisfies an unsupported custom expectation"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def expected(self) -> dict[str, JsonValue]:
        return {"column": self.column}


@pytest.mark.integration
def test_custom_check_subclass_raises_type_error_instead_of_silent_result(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame([(1,)], "value long")
    suite = QualitySuite("unsupported", [_UnsupportedCheck(column="value")])

    with pytest.raises(TypeError, match="Unsupported check type"):
        suite.validate(dataframe)


@pytest.mark.integration
def test_unique_reports_excess_duplicates_not_all_duplicate_group_rows(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame([("A",), ("A",), ("A",), (None,), (None,)], "value string")

    result = QualitySuite("unique", [Unique("value")]).validate(dataframe).results[0]

    assert result.status is CheckStatus.FAIL
    assert result.affected_rows == 2
    assert result.evaluated_rows == 3
    assert result.failure_fraction == pytest.approx(2 / 3)


@pytest.mark.integration
def test_allowed_values_supports_decimal_without_coercion(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(Decimal("1.10"),), (Decimal("2.20"),), (None,)],
        StructType([StructField("value", DecimalType(10, 2))]),
    )

    result = (
        QualitySuite(
            "decimal-allowed",
            [AllowedValues("value", {Decimal("1.10"), Decimal("2.20")})],
        )
        .validate(dataframe)
        .results[0]
    )

    assert result.status is CheckStatus.PASS
    assert result.evaluated_rows == 2
    assert result.failure_fraction == 0.0


@pytest.mark.integration
def test_unique_rejects_map_columns_without_poisoning_valid_checks(
    spark: SparkSession,
) -> None:
    schema = StructType(
        [
            StructField("id", LongType()),
            StructField("attributes", MapType(StringType(), StringType())),
        ]
    )
    dataframe = spark.createDataFrame([(1, {"a": "b"}), (2, {"c": "d"})], schema)
    suite = QualitySuite(
        "map-isolation",
        [NotNull("id"), Unique("attributes"), RowCount(min_count=2)],
    )

    report = suite.validate(dataframe)

    assert [result.status for result in report.results] == [
        CheckStatus.PASS,
        CheckStatus.ERROR,
        CheckStatus.PASS,
    ]
    assert report.metadata.aggregate_actions == 1


@pytest.mark.integration
def test_jvm_regex_semantics_and_malformed_pattern_isolation(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame(
        [(1, "foo.bar"), (2, "other"), (3, ""), (4, None)],
        "id long, text string",
    )
    suite = QualitySuite(
        "regex",
        [
            NotNull("id"),
            MatchesRegex("text", r"\Qfoo.bar\E"),
            MatchesRegex("text", "["),
            MatchesRegex("text", r"^$"),
            RowCount(min_count=4),
        ],
    )

    report = suite.validate(dataframe)

    assert [result.status for result in report.results] == [
        CheckStatus.PASS,
        CheckStatus.FAIL,
        CheckStatus.ERROR,
        CheckStatus.FAIL,
        CheckStatus.PASS,
    ]
    assert report.results[1].affected_rows == 2
    assert report.results[1].evaluated_rows == 3
    assert report.results[2].diagnostic is not None
    assert report.results[2].diagnostic.code == "INVALID_REGEX"
    assert report.metadata.aggregate_actions == 1


@pytest.mark.integration
def test_schema_only_suite_runs_no_spark_action_and_has_phase_timings(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame([], "id long")

    report = QualitySuite("schema-only", [ColumnExists("id"), HasType("id", LongType())]).validate(
        dataframe
    )

    assert report.metadata.aggregate_actions == 0
    assert report.metadata.spark_execution_duration_ms == 0.0
    assert report.metadata.planning_duration_ms >= 0.0
    assert report.metadata.result_evaluation_duration_ms >= 0.0
    assert "execution_duration_ms" not in report.results[0].model_dump()


@pytest.mark.integration
def test_validation_does_not_stop_or_replace_callers_session(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], "id long")
    original_session = dataframe.sparkSession

    QualitySuite("lifecycle", [NotNull("id")]).validate(dataframe)

    assert dataframe.sparkSession is original_session
    assert spark.range(1).count() == 1
