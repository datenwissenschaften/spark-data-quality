"""Construction-time validation tests."""

import math
from collections.abc import Callable
from decimal import Decimal

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
        (lambda: HasType("id", object()), "concrete"),  # type: ignore[arg-type]
        (lambda: InRange("age"), "at least one"),
        (lambda: InRange("age", min_value=2, max_value=1), "must not exceed"),
        (lambda: InRange("age", min_value=math.nan), "finite"),
        (lambda: InRange("age", min_value=math.inf), "finite"),
        (lambda: InRange("age", min_value=Decimal("NaN")), "finite"),
        (lambda: InRange("age", min_value=True), "finite"),
        (lambda: InRange("age", min_value="1"), "finite"),  # type: ignore[arg-type]
        (lambda: AllowedValues("country", frozenset()), "at least one"),
        (lambda: AllowedValues("value", frozenset({1, "1"})), "same scalar type"),
        (lambda: AllowedValues("value", frozenset({math.inf})), "finite"),
        (lambda: MatchesRegex("name", ""), "non-empty"),
        (lambda: RowCount(), "at least one"),
        (lambda: RowCount(min_count=-1), "non-negative"),
        (lambda: RowCount(max_count=-1), "non-negative"),
        (lambda: RowCount(min_count=True), "integer"),
        (lambda: RowCount(min_count=1.5), "integer"),  # type: ignore[arg-type]
        (lambda: RowCount(min_count=2, max_count=1), "must not exceed"),
    ],
)
def test_invalid_check_configuration_fails_early(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_suite_configuration_is_immutable_and_validated() -> None:
    check = HasType("id", IntegerType())
    suite = QualitySuite("players", [check])

    assert suite.checks == (check,)
    with pytest.raises(ValueError, match="name"):
        QualitySuite("", [check])
    with pytest.raises(ValueError, match="at least one"):
        QualitySuite("empty", [])
    with pytest.raises(TypeError, match="Check instances"):
        QualitySuite("wrong-type", ["not-a-check"])  # type: ignore[list-item]


def test_decimal_range_bounds_are_preserved() -> None:
    lower = Decimal("0.123456789012345678")
    check = InRange("value", min_value=lower)

    assert check.min_value is lower
    assert check.expected()["min_value"] == lower


def test_allowed_values_deduplicates_configuration() -> None:
    check = AllowedValues("value", {1, 2})

    assert check.values == frozenset({1, 2})
    assert check.expected()["allowed_values"] == [1, 2]


def test_suite_rejects_identifier_collisions() -> None:
    with pytest.raises(ValueError, match="identifiers"):
        QualitySuite(
            "duplicate-ids",
            [ColumnExists("a", check_id="same"), ColumnExists("b", check_id="same")],
        )

    with pytest.raises(ValueError, match="identifiers"):
        QualitySuite(
            "generated-collision",
            [ColumnExists("a", check_id="column_exists:1"), ColumnExists("b")],
        )
