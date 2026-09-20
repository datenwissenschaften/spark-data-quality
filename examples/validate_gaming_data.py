"""Validate and profile a small deterministic player dataset."""

from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from spark_data_quality import (
    AllowedValues,
    ColumnExists,
    HasType,
    InRange,
    NotNull,
    QualitySuite,
    Unique,
    profile,
)


def main() -> None:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("quality-example")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    try:
        players = spark.createDataFrame(
            [(1, 24, "DE"), (2, 31, "ZA"), (3, 19, "GB")],
            "player_id long, age long, country string",
        )
        suite = QualitySuite(
            "players",
            [
                ColumnExists("player_id"),
                HasType("player_id", LongType()),
                NotNull("player_id"),
                Unique("player_id"),
                InRange("age", min_value=13, max_value=120),
                AllowedValues("country", frozenset({"DE", "ZA", "GB"})),
            ],
        )
        print(suite.validate(players).model_dump_json(indent=2))
        print(profile(players, ["age", "country"], top_k=3).model_dump_json(indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
