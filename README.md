# spark-data-quality

`spark-data-quality` is a small, typed validation library for PySpark DataFrames. It
turns reusable expectations into Spark expressions, combines compatible metrics into
one aggregation, and returns Pydantic reports that remain usable after the Spark
session ends.

The library is intended for batch pipelines, local Spark applications, and managed
Spark runtimes such as Databricks. It has no Databricks-specific runtime dependency.

See the end-to-end data-quality workflow →
[`notebooks/data_quality_workflow.ipynb`](notebooks/data_quality_workflow.ipynb), a
game-analytics case study that profiles, validates, and visualizes a synthetic
player/session dataset with this library.

![Quality check outcomes for the notebook's synthetic game_sessions_ingest suite](docs/assets/data-quality-workflow/quality-results.png)

## Example

```python
from pyspark.sql import SparkSession

from spark_data_quality import (
    AllowedValues,
    ColumnExists,
    InRange,
    NotNull,
    QualitySuite,
    Unique,
)

spark = SparkSession.builder.getOrCreate()
players = spark.createDataFrame(
    [(1, 24, "DE"), (2, 31, "ZA")],
    "player_id long, age long, country string",
)

suite = QualitySuite(
    name="players",
    checks=[
        ColumnExists("player_id"),
        NotNull("player_id"),
        Unique("player_id"),
        InRange("age", min_value=13, max_value=120),
        AllowedValues("country", frozenset({"DE", "ZA", "GB"})),
    ],
)

report = suite.validate(players)
print(report.status)
print(report.model_dump_json(indent=2))
```

## Core capabilities

- Immutable, typed expectation definitions with construction-time configuration checks
- Explicit `PASS`, `FAIL`, and `ERROR` outcomes
- Spark SQL expressions and aggregations only; no Python UDFs or pandas conversion
- One shared aggregate action for compatible validation metrics
- Deterministic check ordering, identifiers, metric aliases, and categorical tie-breaking
- Standards-compatible JSON reports with row counts, explicit denominators, and diagnostics
- Numeric and string profiling with bounded top-value collection

## Architecture

```mermaid
flowchart LR
    A[QualitySuite and typed checks] --> B[AggregationPlanner]
    B --> C[Schema-only results]
    B --> D[Shared aggregate expressions]
    D --> E[Spark DataFrame.agg]
    E --> F[One scalar Row on driver]
    C --> G[Result evaluator]
    F --> G
    G --> H[Pydantic SuiteReport]
```

Check objects define expectations and configuration. The planner resolves schema
requirements and deduplicates scalar metrics. The runner executes the aggregate plan.
The evaluator interprets scalar values and constructs public result models. See
[the architecture notes](docs/architecture.md) for the boundaries and error semantics.

## Execution model

`ColumnExists` and `HasType` use the DataFrame schema and require no action. All valid
row-level checks in a suite contribute expressions to one `DataFrame.agg` action. The
planner deduplicates identical total, null, and other metrics. Exact uniqueness uses
`count` and `countDistinct` in that aggregate; Spark may introduce a shuffle and
multiple physical stages even though the library starts one aggregate action.

The validation runner collects exactly one aggregate row. It never collects source
rows or unbounded values. Execution metadata reports the number of library-level
aggregate actions and scalar expressions; it does not claim to count Spark stages.
It also separates suite planning, shared Spark execution, result evaluation, and total
wall-clock durations. Individual checks do not claim a private share of the common
Spark execution time.

The profiler similarly shares scalar statistics in one aggregate action. Each selected
string column needs a separate grouped top-values action. Those results are ordered by
count descending and value ascending, then limited to `top_k` (maximum 100).

## Available checks

| Check | Semantics | SQL NULL | Floating-point NaN | Empty DataFrame |
| --- | --- | --- | --- | --- |
| `ColumnExists` | Exact top-level name is present | Not applicable | Not applicable | Uses schema |
| `HasType` | Exact Spark `DataType` equality | Type is schema-level | Type is schema-level | Uses schema |
| `NotNull` | SQL null count is zero | Fails | Not a null; passes | Passes vacuously |
| `Unique` | Exact distinct count equals non-null count | Ignored, including repeated nulls | Participates as a value; repeated NaNs are duplicates | Passes vacuously |
| `InRange` | Values satisfy configured inclusive or exclusive bounds | Ignored | Always fails | Passes vacuously |
| `AllowedValues` | Values belong to a non-empty, homogeneous scalar set | Ignored | Fails because configured values must be finite | Passes vacuously |
| `MatchesRegex` | Spark/JVM `rlike` matches the pattern | Ignored | Non-string columns are `ERROR` | Passes vacuously |
| `RowCount` | Count satisfies inclusive minimum/maximum bounds | Not applicable | Not applicable | Depends on bounds |

An absent column produces `FAIL` for `ColumnExists` and `ERROR` for checks that need
that column to continue. A `HasType` mismatch is a valid evaluation and therefore
produces `FAIL`. `InRange`, `MatchesRegex`, and `AllowedValues` reject
incompatible Spark types with `ERROR`. Use `NotNull` beside checks that should also
reject nulls.

`HasType` compares Spark `DataType` objects directly. It does not group byte, short,
integer, and long columns into one logical family, and decimal precision and scale are
part of equality.

For `Unique`, `affected_rows` is duplicate excess: non-null row count minus exact
distinct count. This is stable and aggregate-friendly; it is not the count of every row
participating in a duplicate group. Spark normalizes NaN for exact distinct aggregation,
so two NaNs contribute one excess duplicate. Multiple nulls never contribute.

For checks that ignore nulls, `evaluated_rows` is the non-null count and
`failure_fraction` is `affected_rows / evaluated_rows`. For `NotNull`, every source row
is evaluated. When zero rows are evaluated, `failure_fraction` is `None` rather than a
fabricated zero. `total_rows` always describes the complete DataFrame when a row-level
check ran.

Range bounds must be finite. NaN is explicitly invalid regardless of one-sided or
two-sided bounds. Positive and negative infinity are compared normally, so they pass
only when the configured finite bound does not exclude them. Decimal columns accept
`Decimal` or integer bounds; float bounds are rejected to avoid silent precision loss.

Regex patterns follow Java's regular-expression syntax because Spark evaluates `rlike`
on the JVM. Empty patterns are rejected at construction. The planner asks the active
Spark JVM to compile other patterns without starting a Spark action; malformed patterns
become an `INVALID_REGEX` result and never enter the shared aggregate.

## Profiling

```python
from spark_data_quality import profile

profile_report = profile(players, ["age", "country"], top_k=5)
print(profile_report.model_dump_json(indent=2))
```

Numeric profiles contain null and distinct counts, minimum, maximum, mean, and sample
standard deviation. Float and double profiles also contain `nan_count`. Spark's native
aggregate semantics are preserved: NaN sorts above finite values and may make maximum,
mean, or standard deviation NaN; infinities can likewise produce non-finite aggregates.
Reports encode non-finite values as `"NaN"`, `"Infinity"`, or `"-Infinity"`, producing
standards-compatible JSON. Decimal min, max, and mean remain `Decimal` values in Python
and serialize without conversion through binary float. `standard_deviation` is the
exception: Spark's `stddev_samp` always returns `DoubleType`, even for a `DecimalType`
input column, so it is reported as `float` while the other statistics for that same
column stay `Decimal`. This is Spark's own aggregate-function typing, not a conversion
performed by this library. String profiles contain null and
distinct counts, minimum and
maximum lengths, and bounded frequent values. Unsupported selected types are rejected
before execution instead of receiving ambiguous statistics.

## Fugue integration (optional)

**Native PySpark is the optimized, primary backend for everything above.** The
optional `fugue` extra adds exactly one thing: the ability to hand `QualitySuite`
or `profile` a non-PySpark, Fugue-compatible dataframe (pandas, Arrow, Polars,
DuckDB, a list of dictionaries, ...) instead of hand-writing the Spark
conversion yourself.

```bash
pip install spark-data-quality[fugue]
```

```python
import pandas as pd

from spark_data_quality import InRange, NotNull, QualitySuite
from spark_data_quality.fugue import validate

suite = QualitySuite("players", [NotNull("player_id"), InRange("age", min_value=13, max_value=120)])
pandas_players = pd.DataFrame({"player_id": [1, 2, 3], "age": [24, 31, 19]})

report = validate(suite, pandas_players)  # binds to SparkSession.builder.getOrCreate() by default
```

`spark_data_quality.fugue` converts the input into a native PySpark `DataFrame`
bound to a caller-controlled `SparkSession` (pass `spark=...` to bind explicitly;
it never creates or stops a session beyond PySpark's own ambient-session
convention), then calls the exact same `QualitySuite.validate` used for native
input. There is no separate Fugue execution engine for checks themselves:

- **Fugue provides input portability**, not alternate execution semantics. Every
  built-in check runs through the identical Spark-native aggregation planning
  described above, whether the input arrived as a PySpark `DataFrame` directly or
  through this conversion. An already-native PySpark `DataFrame` passes through
  unchanged (no conversion cost on the optimized path).
- Because there is only one execution path (Spark) behind this boundary, check
  semantics are identical for every built-in check regardless of the input
  source — see the table below.
- Converting *from* pandas has real, verified caveats independent of this
  library: a pandas float `NaN` becomes Spark `NaN` (not SQL `NULL`), a pandas
  integer column with a missing value is upcast to `float64` before Fugue ever
  sees it, and a null inside a schema-declared integral column currently fails
  the conversion outright with `PySparkTypeError` rather than silently
  succeeding. These are documented in `spark_data_quality/fugue.py` and
  exercised by `tests/integration/test_fugue.py`; pass an explicit Fugue
  `schema` string, or use a native PySpark `DataFrame`, when a column's exact
  type or null semantics must be preserved.

| Check | Native PySpark | Via `spark_data_quality.fugue` | Notes |
| --- | --- | --- | --- |
| `ColumnExists` | yes | yes (identical) | Executes as native Spark after conversion. |
| `HasType` | yes | yes (identical) | Sensitive to the pandas type-inference caveats above. |
| `NotNull` | yes | yes (identical) | Pandas `NaN` is not SQL `NULL`; see caveats above. |
| `Unique` | yes | yes (identical) | No behavior change; input source is irrelevant post-conversion. |
| `InRange` | yes | yes (identical) | Explicitly rejects NaN regardless of input source. |
| `AllowedValues` | yes | yes (identical) | No behavior change. |
| `MatchesRegex` | yes | yes (identical) | No behavior change. |
| `RowCount` | yes | yes (identical) | No behavior change. |

The `notebook` extra's dependencies (Jupyter, Plotly, Kaleido, pandas) are
required only to reproduce `notebooks/data_quality_workflow.ipynb`; neither
`fugue` nor `notebook` is required to use the core library. Installing `fugue`
pulls in `fugue[spark]`, and transitively pandas, pyarrow, and a few smaller
packages; `packaging` is pinned explicitly in this extra as a safeguard for a
verified gap in `fugue`'s own dependency (`triad` imports `packaging` without
declaring it), which otherwise surfaces as `ModuleNotFoundError` in a bare
environment that installs only this extra's declared requirements.

## Installation

Python 3.12, PySpark 4.0 through 4.1, and a compatible JVM are required. CI uses Java 17.

```bash
pip install spark-data-quality
```

For source development with Poetry:

```bash
poetry install --with dev
poetry install --with dev --extras "fugue notebook"   # optional integrations
```

## Development and testing

The test suite starts a real `local[2]` Spark session. It requires no cloud services,
credentials, or network access after dependencies are installed.

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Regenerate the end-to-end notebook and its committed PNG assets (requires the
`fugue` and `notebook` extras):

```bash
make notebooks
```

## Limitations

- Exact `countDistinct` can cause an expensive shuffle on high-cardinality columns.
- Large `AllowedValues` sets become literal `isin` expressions; use a reference-table
  join outside this library when the allowed domain contains thousands of values.
- The profiler supports numeric and string columns only.
- Top-value profiling performs one bounded grouped action per selected string column.
- Validation reports aggregate failures but do not include failing source-row samples.
- JVM regex prevalidation uses the classic PySpark JVM gateway. Spark Connect is not a
  currently supported execution mode.
- Durations are wall-clock observations and are not stable benchmark measurements.
- `Check` is a public abstract base class, but the planner and evaluator only recognize
  the built-in check types above. Constructing a suite with a custom `Check` subclass
  raises `TypeError` from `QualitySuite.validate` rather than silently skipping the
  check or producing a fabricated result; this is deliberate (see
  [Architecture](docs/architecture.md)) but means user-defined checks are not currently
  a supported extension point.
- `spark_data_quality.fugue` executes checks on Spark only. It does not run checks
  natively on pandas, DuckDB, Polars, or any other Fugue-supported engine; it converts
  the input into Spark first. A null inside a schema-declared integral column does not
  currently survive Fugue's row-oriented (list-of-dictionaries) conversion path — it
  fails with `PySparkTypeError` rather than being silently dropped or reinterpreted.

## Roadmap

Technically useful next steps include configurable approximate distinct counts,
cross-column predicates, bounded failing-row samples with explicit privacy controls,
and a second execution group for checks that cannot share scalar aggregation. A CLI is
not planned until it provides value beyond the Python API.

## License

MIT
