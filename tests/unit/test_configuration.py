"""Construction-time validation tests."""

import math

import pytest
from pyspark.sql.types import DataType, IntegerType

from spark_data_quality import (
    AllowedValues,
    ColumnExists,
    HasType,
    InRange,
    MatchesRegex,
    QualitySuite,
    RowCount,
)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ColumnExists(""), "column"),
        (lambda: ColumnExists("id", check_id=" "), "check_id"),
        (lambda: HasType("id", DataType()), "concrete"),
        (lambda: InRange("age"), "at least one"),
        (lambda: InRange("age", min_value=2, max_value=1), "must not exceed"),
        (lambda: InRange("age", min_value=math.nan), "finite"),
        (lambda: AllowedValues("country", frozenset()), "at least one"),
        (lambda: AllowedValues("value", frozenset({1, "1"})), "same scalar type"),
        (lambda: MatchesRegex("name", "["), "valid Python"),
        (lambda: RowCount(), "at least one"),
        (lambda: RowCount(min_count=-1), "non-negative"),
        (lambda: RowCount(min_count=2, max_count=1), "must not exceed"),
    ],
)
def test_invalid_check_configuration_fails_early(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_suite_configuration_is_immutable_and_validated() -> None:
    check = HasType("id", IntegerType())
    suite = QualitySuite("players", [check])

    assert suite.checks == (check,)
    with pytest.raises(ValueError, match="name"):
        QualitySuite("", [check])
    with pytest.raises(ValueError, match="at least one"):
        QualitySuite("empty", [])
