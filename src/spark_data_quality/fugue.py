"""Optional Fugue integration: portable dataframe input into the native Spark backend.

This module requires the ``fugue`` extra (``pip install spark-data-quality[fugue]``).
Importing :mod:`spark_data_quality` itself never imports Fugue; only importing this
module does.

## Integration boundary

Fugue's own execution abstraction (dataframe conversion between pandas, Arrow,
Polars, DuckDB, and other engines it recognizes) is used for exactly one purpose
here: materializing a caller-supplied, Fugue-compatible dataframe into a native
PySpark ``DataFrame`` bound to a caller-controlled ``SparkSession``. Once that
conversion happens, every check, the ``AggregationPlanner``, the shared
``DataFrame.agg`` execution, and the result/report models are the exact same
code path used by :meth:`spark_data_quality.suite.QualitySuite.validate` for a
native PySpark input. An already-native PySpark ``DataFrame`` passes through
this boundary unchanged (Fugue treats it as a no-op conversion), so the native
path pays no Fugue overhead and its aggregation-planning guarantees are
untouched.

This module intentionally does *not* execute checks on non-Spark Fugue engines
(pandas, DuckDB, Polars, ...). Doing so would require re-implementing every
check's Spark SQL type semantics, JVM regex evaluation, and NaN/NULL/Decimal
handling per engine, which risks silently producing different data-quality
answers on different backends -- exactly the "fake portability" this project
avoids. Fugue therefore provides *input* portability (accept dataframes from
many sources without hand-written Spark conversion code) rather than
*execution* portability (run checks identically on many engines).

## Known semantic caveats when converting from pandas

Fugue converts many source types through pandas/Arrow. Two pandas conventions
do not map onto this library's documented Spark NULL/NaN/type semantics:

- A pandas ``float64`` column's ``NaN`` sentinel (however it originated, including
  a missing value in the source data) becomes Spark ``NaN``, not SQL ``NULL``.
  ``NotNull`` therefore does not catch it; a check that explicitly rejects NaN
  (for example ``InRange``) is required if that distinction matters.
- A pandas integer column containing a missing value is silently upcast by
  pandas itself (before Fugue ever sees it) to ``float64``, so it arrives in
  Spark as ``DoubleType`` rather than ``LongType`` or ``IntegerType``. A
  ``HasType`` check against an integral type will therefore fail even though
  the values themselves look integral.
- An explicit Fugue ``schema`` string (for example ``"player_id:long"``)
  fixes this for a column with **no** missing values: it bypasses pandas type
  inference and produces the declared Spark type directly. It does **not**
  help when that column also has a null: Fugue's row-oriented conversion path
  (used for a list of dictionaries, for example) still materializes the data
  through pandas internally, and pandas has no representation for a null in a
  plain integer column, so the conversion raises ``PySparkTypeError`` instead
  of silently downgrading the type or dropping the null. A native Spark
  ``DataFrame`` with an explicit schema is the reliable way to combine an
  integral type with real nulls.

These are consequences of pandas' own type system interacting with Fugue's
conversion path, not decisions this integration makes; they are exercised by
`tests/integration/test_fugue.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

try:
    import fugue.api as _fugue_api
except ModuleNotFoundError as error:  # pragma: no cover - exercised via subprocess test
    raise ImportError(
        "spark_data_quality.fugue requires the optional 'fugue' extra. "
        "Install it with: pip install spark-data-quality[fugue]"
    ) from error

try:
    from fugue_spark import SparkExecutionEngine
except ModuleNotFoundError as error:  # pragma: no cover - exercised via subprocess test
    raise ImportError(
        "spark_data_quality.fugue requires Fugue's Spark execution backend. "
        "Install it with: pip install spark-data-quality[fugue]"
    ) from error

from spark_data_quality.models import SuiteReport
from spark_data_quality.profiling import ProfileReport, profile
from spark_data_quality.suite import QualitySuite


def to_spark_dataframe(
    dataframe: Any,
    *,
    schema: Any = None,
    spark: SparkSession | None = None,
) -> SparkDataFrame:
    """Materialize any Fugue-compatible dataframe as a native PySpark ``DataFrame``.

    ``dataframe`` may already be a PySpark ``DataFrame`` (returned unchanged),
    or any other type Fugue recognizes for its Spark execution engine, such as
    a pandas ``DataFrame``, a ``pyarrow.Table``, a Polars ``DataFrame``, or a
    list of dictionaries paired with an explicit ``schema``.

    ``spark`` binds the conversion to a caller-owned ``SparkSession``. This
    function never creates or stops a session itself: when ``spark`` is
    omitted, Fugue's ``SparkExecutionEngine`` falls back to
    ``SparkSession.builder.getOrCreate()``, the same ambient-session
    convention PySpark itself uses. Ownership and lifecycle of that session
    remain the caller's responsibility.

    ``schema`` is a Fugue schema expression (for example
    ``"player_id:long,country:str"``). It is required for schema-less inputs
    such as a list of dictionaries, and it is the recommended way to avoid the
    pandas type-inference caveats documented in this module's docstring.
    """

    engine = SparkExecutionEngine(spark_session=spark)
    # fugue.api re-exports these without an `__all__`, so mypy's strict
    # no_implicit_reexport check cannot see them as public; both are documented
    # top-level entry points of fugue.api.
    engine_dataframe = _fugue_api.as_fugue_engine_df(  # type: ignore[attr-defined]
        engine, dataframe, schema=schema
    )
    native = _fugue_api.get_native_as_df(  # type: ignore[attr-defined]
        engine_dataframe
    )
    if not isinstance(native, SparkDataFrame):
        raise TypeError(
            "Fugue's Spark execution engine returned a "
            f"{type(native).__name__!r} dataframe instead of a PySpark DataFrame; "
            "this indicates an unsupported Fugue engine configuration."
        )
    return native


def validate(
    suite: QualitySuite,
    dataframe: Any,
    *,
    schema: Any = None,
    spark: SparkSession | None = None,
) -> SuiteReport:
    """Validate a Fugue-compatible dataframe with the native Spark-optimized suite.

    This performs exactly one conversion at the Fugue boundary
    (`to_spark_dataframe`) and then calls
    :meth:`~spark_data_quality.suite.QualitySuite.validate` unmodified: the
    same aggregation planning, PASS/FAIL/ERROR semantics, and result model as
    validating a native PySpark ``DataFrame`` directly. There is no separate,
    Fugue-specific check execution path.
    """

    return suite.validate(to_spark_dataframe(dataframe, schema=schema, spark=spark))


def profile_dataframe(
    dataframe: Any,
    columns: Sequence[str] | None = None,
    *,
    top_k: int = 10,
    schema: Any = None,
    spark: SparkSession | None = None,
) -> ProfileReport:
    """Profile a Fugue-compatible dataframe using the native Spark-native profiler.

    Converts ``dataframe`` at the Fugue boundary (`to_spark_dataframe`) and
    then delegates to :func:`spark_data_quality.profiling.profile` unmodified.
    """

    return profile(
        to_spark_dataframe(dataframe, schema=schema, spark=spark),
        list(columns) if columns is not None else None,
        top_k=top_k,
    )


__all__ = ["profile_dataframe", "to_spark_dataframe", "validate"]
