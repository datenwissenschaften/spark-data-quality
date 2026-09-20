"""Serializable scalar and recursive value types used by public reports."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


class NonFiniteFloat(StrEnum):
    """Standards-compatible representation of non-finite floating-point values."""

    NAN = "NaN"
    POSITIVE_INFINITY = "Infinity"
    NEGATIVE_INFINITY = "-Infinity"


type JsonScalar = None | bool | int | float | str | Decimal | date | datetime
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type NumericValue = int | float | Decimal | NonFiniteFloat
