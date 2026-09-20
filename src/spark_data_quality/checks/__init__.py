"""Built-in data-quality checks."""

from spark_data_quality.checks.base import Check
from spark_data_quality.checks.builtin import (
    AllowedValues,
    ColumnExists,
    HasType,
    InRange,
    MatchesRegex,
    NotNull,
    RowCount,
    Unique,
)

__all__ = [
    "AllowedValues",
    "Check",
    "ColumnExists",
    "HasType",
    "InRange",
    "MatchesRegex",
    "NotNull",
    "RowCount",
    "Unique",
]
