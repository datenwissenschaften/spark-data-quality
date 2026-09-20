"""Public library exceptions."""


class SparkDataQualityError(Exception):
    """Base exception for library-level errors."""


class InvalidProfileError(SparkDataQualityError, ValueError):
    """Raised when a requested profile cannot be planned safely."""
