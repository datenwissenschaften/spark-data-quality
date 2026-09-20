"""Consistency validation for profiling report models."""

import pytest
from pydantic import ValidationError

from spark_data_quality.profiling.models import ColumnProfile, NumericStatistics, StringStatistics

_NUMERIC_STATS = NumericStatistics(minimum=0, maximum=1, mean=0.5, standard_deviation=0.1)
_STRING_STATS = StringStatistics(minimum_length=1, maximum_length=1, top_values=())


def test_numeric_profile_rejects_missing_numeric_statistics() -> None:
    with pytest.raises(ValidationError, match="numeric profiles require only numeric statistics"):
        ColumnProfile(
            column="value",
            spark_type="bigint",
            kind="numeric",
            null_count=0,
            distinct_count=0,
        )


def test_numeric_profile_rejects_string_statistics() -> None:
    with pytest.raises(ValidationError, match="numeric profiles require only numeric statistics"):
        ColumnProfile(
            column="value",
            spark_type="bigint",
            kind="numeric",
            null_count=0,
            distinct_count=0,
            numeric=_NUMERIC_STATS,
            string=_STRING_STATS,
        )


def test_string_profile_rejects_missing_string_statistics() -> None:
    with pytest.raises(ValidationError, match="string profiles require only string statistics"):
        ColumnProfile(
            column="value",
            spark_type="string",
            kind="string",
            null_count=0,
            distinct_count=0,
        )


def test_string_profile_rejects_numeric_statistics() -> None:
    with pytest.raises(ValidationError, match="string profiles require only string statistics"):
        ColumnProfile(
            column="value",
            spark_type="string",
            kind="string",
            null_count=0,
            distinct_count=0,
            numeric=_NUMERIC_STATS,
            string=_STRING_STATS,
        )
