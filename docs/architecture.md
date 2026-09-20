# Architecture

## Design boundaries

The implementation keeps four concerns separate:

1. A `Check` is an immutable expectation definition. It validates its own configuration
   but does not hold a DataFrame or execute Spark.
2. `AggregationPlanner` resolves column and type requirements against the schema. It
   creates deterministic scalar metric expressions and deduplicates compatible metrics.
3. `execute_plan` submits the shared aggregation and copies the one scalar result row
   into Python values.
4. `evaluate_check` interprets those scalar values and creates Pydantic results.

This split makes configuration testable without Spark, keeps cost decisions visible,
and prevents Spark `Column`, schema, session, or context objects from leaking into
serialized reports.

## Check abstraction

Every check exposes a stable `check_type`, description, required columns, and a
JSON-compatible expected constraint. Checks are frozen dataclasses. A caller may assign
a stable `check_id`; otherwise the suite derives one from check type and ordered index.

Construction rejects invalid configurations such as empty allowed sets, mixed allowed
value types, invalid regular expressions, non-finite range bounds, and contradictory
row-count bounds. Conditions that depend on a DataFrame schema are resolved by the
planner.

## Aggregation planning

The planner stores metrics by a semantic key such as `(null_count, column, arguments)`.
If checks request the same metric, they receive the same deterministic alias. Valid
row-level checks also share the total row count. Expressions currently produced are:

- `count(1)` for total rows;
- conditional `sum` expressions for null or invalid counts;
- `count(column)` and exact `countDistinct(column)` for uniqueness;
- conditional comparisons, `isin`, and `rlike` for built-in predicates.

All expressions are submitted in one `DataFrame.agg` call. This is one library-level
Spark action and one logical aggregate query. Exact distinct aggregation can still make
Spark produce shuffle exchanges and multiple execution stages. The metadata deliberately
reports aggregate actions and expression counts, not an inferred number of Spark jobs or
stages.

Schema checks have no aggregate metrics. A suite containing only schema checks performs
no Spark action.

## Spark execution

The executor calls `first()` on the aggregate DataFrame and copies that single `Row` to
a dictionary. Source data is never converted to pandas, iterated in Python, or collected
to the driver. Spark SQL expressions are used instead of Python UDFs.

Planning errors that are predictable from the schema become result diagnostics. Spark
execution exceptions are allowed to propagate: an unavailable cluster, invalid runtime
state, or Spark engine failure is different from a data-quality result and should retain
its actionable exception.

## Result model

`SuiteReport` contains ordered `CheckResult` values and `ExecutionMetadata`. Fields such
as affected rows, total rows, and failure fraction are optional because they do not
apply to schema checks. All public report objects are frozen Pydantic models and support
`model_dump`, `model_dump_json`, and normal Pydantic validation.

Individual data-check duration includes the shared aggregation wall time. It should not
be summed across checks because each check refers to the same shared action. Suite total
duration is the end-to-end wall time.

Suite status is deterministic:

1. `ERROR` if any check is `ERROR`;
2. otherwise `FAIL` if any check is `FAIL`;
3. otherwise `PASS`.

## PASS, FAIL, and ERROR

`PASS` means a valid expectation was evaluated and satisfied. `FAIL` means a valid
expectation was evaluated and the observed data did not satisfy it. A Spark type that
does not equal `HasType.expected_type` is therefore `FAIL`.

`ERROR` means evaluation cannot validly answer the expectation. Missing required
columns for dependent checks and incompatible input types are current examples. A
missing column is `FAIL` for `ColumnExists`, because that check can answer its
expectation directly. An `ERROR` contains a
bounded diagnostic code, message, column, and small details mapping; it never embeds a
Spark stack trace.

Programming defects and unexpected Spark failures are not converted to a generic
`ERROR`. They propagate so pipeline infrastructure and developers can handle the real
failure.

## Null and empty-input semantics

`NotNull` is the only built-in value check that treats null as a violation. `Unique`,
`InRange`, `AllowedValues`, and `MatchesRegex` ignore nulls and compose with `NotNull`.
This avoids making one check silently imply another expectation.

On an empty DataFrame, all value checks pass with zero affected rows and a failure
fraction of `0.0`. `ColumnExists` and `HasType` still use the declared schema. `RowCount`
passes or fails according to its configured bounds.

## Profiling execution

Profiling builds one shared scalar aggregate for all selected numeric and string
columns. It collects one scalar row. Frequent string values require grouped queries;
each query excludes null, uses deterministic ordering, applies `limit(top_k)`, and only
then collects. `top_k` is capped at 100, so collected category output is bounded even
for high-cardinality columns.
