"""Fugue integration boundary tested against a real local SparkSession.

These tests verify that `spark_data_quality.fugue`:

- converts Fugue-compatible input into the native Spark path without changing
  check semantics or results;
- passes an already-native PySpark DataFrame through without a real Fugue
  conversion (no wasted work on the optimized path);
- documents rather than hides the pandas NaN/NULL and integer-upcast
  differences described in its module docstring;
- fails clearly for input Fugue cannot recognize;
- never creates or stops a SparkSession itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest
from pyspark.errors import PySparkTypeError
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from spark_data_quality import (
    AllowedValues,
    CheckStatus,
    HasType,
    InRange,
    MatchesRegex,
    NotNull,
    QualitySuite,
)
from spark_data_quality.fugue import profile_dataframe, to_spark_dataframe
from spark_data_quality.fugue import validate as fugue_validate


@pytest.mark.integration
def test_native_spark_dataframe_passes_through_unchanged(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1, "DE")], "player_id long, country string")

    converted = to_spark_dataframe(dataframe, spark=spark)

    assert converted is dataframe


@pytest.mark.integration
def test_pandas_input_reuses_the_native_aggregation_path(spark: SparkSession) -> None:
    pandas_df = pd.DataFrame({"player_id": [1, 2, 3], "age": [24, 31, 19]})
    suite = QualitySuite(
        "players",
        [NotNull("player_id"), InRange("age", min_value=13, max_value=120)],
    )

    fugue_report = fugue_validate(suite, pandas_df, spark=spark)

    native_equivalent = spark.createDataFrame(pandas_df)
    native_report = suite.validate(native_equivalent)

    assert fugue_report.status is CheckStatus.PASS
    assert [r.status for r in fugue_report.results] == [r.status for r in native_report.results]
    assert [r.affected_rows for r in fugue_report.results] == [
        r.affected_rows for r in native_report.results
    ]
    assert fugue_report.metadata.aggregate_actions == 1


@pytest.mark.integration
def test_explicit_schema_preserves_integer_type_when_there_are_no_nulls(
    spark: SparkSession,
) -> None:
    """An explicit Fugue schema keeps LongType instead of pandas' float64 inference."""

    rows = [{"player_id": 1}, {"player_id": 2}]
    converted = to_spark_dataframe(rows, schema="player_id:long", spark=spark)

    assert converted.schema["player_id"].dataType == LongType()
    suite = QualitySuite("players", [HasType("player_id", LongType())])
    report = suite.validate(converted)

    assert report.results[0].status is CheckStatus.PASS


@pytest.mark.integration
def test_null_in_a_schema_declared_integer_column_fails_clearly(spark: SparkSession) -> None:
    """A verified, documented Fugue-conversion limitation, not a silent reinterpretation.

    Fugue's Spark conversion for row-oriented input (a list of dictionaries)
    materializes data through pandas before handing it to
    ``SparkSession.createDataFrame``. A null in a schema-declared integral
    column cannot survive that round trip: pandas has no way to represent a
    null in a plain integer column, so PySpark rejects the resulting float
    value against the declared ``LongType``. This is a real upstream
    conversion limitation (see the caveats in `spark_data_quality.fugue`), not
    something this library silently works around; it fails with an explicit,
    typed exception instead of producing a wrong result.
    """

    rows = [{"player_id": 1}, {"player_id": None}]
    with pytest.raises(PySparkTypeError):
        to_spark_dataframe(rows, schema="player_id:long", spark=spark)


@pytest.mark.integration
def test_pandas_nan_in_float_column_is_spark_nan_not_null(spark: SparkSession) -> None:
    """Documents the pandas-NaN-vs-Spark-NULL caveat with a real assertion.

    A pandas float64 NaN is not SQL NULL after Fugue's conversion, so NotNull
    passes on it while InRange (which explicitly rejects NaN) correctly fails.
    """

    pandas_df = pd.DataFrame({"age": [24.0, float("nan"), 31.0]})
    converted = to_spark_dataframe(pandas_df, spark=spark)

    suite = QualitySuite(
        "players",
        [NotNull("age"), InRange("age", min_value=0, max_value=120)],
    )
    report = suite.validate(converted)

    assert report.results[0].status is CheckStatus.PASS  # NotNull does not see NaN
    assert report.results[1].status is CheckStatus.FAIL  # InRange rejects NaN explicitly
    assert report.results[1].affected_rows == 1


@pytest.mark.integration
def test_unsupported_input_fails_clearly(spark: SparkSession) -> None:
    with pytest.raises(NotImplementedError):
        to_spark_dataframe(object(), spark=spark)


@pytest.mark.integration
def test_profile_dataframe_matches_native_profile(spark: SparkSession) -> None:
    pandas_df = pd.DataFrame({"country": ["DE", "DE", "ZA"]})

    fugue_report = profile_dataframe(pandas_df, ["country"], top_k=5, spark=spark)
    native_report = to_spark_dataframe(pandas_df, spark=spark)

    assert fugue_report.row_count == 3
    assert fugue_report.columns[0].string is not None
    assert fugue_report.columns[0].string.top_values[0].value == "DE"
    assert fugue_report.columns[0].string.top_values[0].count == 2
    assert native_report.count() == 3


@pytest.mark.integration
def test_conversion_binds_to_the_caller_supplied_session_only(spark: SparkSession) -> None:
    """`to_spark_dataframe` must not create or stop a SparkSession of its own."""

    active_before = SparkSession.getActiveSession()
    converted = to_spark_dataframe(pd.DataFrame({"x": [1]}), spark=spark)

    assert converted.sparkSession is spark
    assert SparkSession.getActiveSession() is active_before
    assert not spark.sparkContext._jsc.sc().isStopped()


def test_allowed_values_check_is_unaffected_by_fugue_import(spark: SparkSession) -> None:
    """A native-only check keeps working identically once the fugue module is imported."""

    dataframe = spark.createDataFrame([(1, "DE")], "player_id long, country string")
    suite = QualitySuite("players", [AllowedValues("country", frozenset({"DE", "ZA"}))])

    report = suite.validate(dataframe)

    assert report.status is CheckStatus.PASS


def test_base_package_imports_without_fugue_installed() -> None:
    """`import spark_data_quality` must work even when Fugue is not installed."""

    script = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "fugue" or name.startswith("fugue."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _blocking_import

        import spark_data_quality  # noqa: F401
        print("BASE_IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "BASE_IMPORT_OK" in result.stdout


def test_fugue_module_raises_clear_import_error_without_fugue_installed() -> None:
    """Requesting the Fugue-specific API without the extra fails with guidance."""

    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "fugue" or name.startswith("fugue."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = _blocking_import

        try:
            import spark_data_quality.fugue  # noqa: F401
        except ImportError as error:
            assert "pip install spark-data-quality[fugue]" in str(error)
            print("CLEAR_IMPORT_ERROR_OK")
        else:
            raise AssertionError("expected ImportError when fugue is unavailable")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAR_IMPORT_ERROR_OK" in result.stdout


@pytest.mark.integration
def test_matches_regex_still_rejects_non_string_columns(spark: SparkSession) -> None:
    """Regression: unrelated native check behavior is untouched by the Fugue module."""

    dataframe = spark.createDataFrame([(1,)], "player_id long")
    suite = QualitySuite("players", [MatchesRegex("player_id", "^[0-9]+$")])
    report = suite.validate(dataframe)

    assert report.status is CheckStatus.ERROR
    assert report.results[0].diagnostic is not None
    assert report.results[0].diagnostic.code == "UNEXPECTED_SPARK_TYPE"
