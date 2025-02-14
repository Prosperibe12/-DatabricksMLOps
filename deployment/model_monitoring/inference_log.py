import json
import pandas as pd
from pyspark.sql.functions import lit, col, from_unixtime, array 
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.sql import DataFrame, types as T, functions as F
from pyspark.sql.functions import current_date, to_date

"""
This script provides functions to process and transform JSON inference logs for model monitoring in a Databricks environment.
Functions:
    convert_to_record_json(json_str: str) -> str:
        Converts various JSON formats into a common JSON format suitable for processing.
        Args:
            json_str (str): The input JSON string.
        Returns:
            str: The converted JSON string in a common format or the original JSON string if unsupported.
    json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
        A pandas UDF that applies the convert_to_record_json function to a series of JSON strings.
        Args:
            json_strs (pd.Series): A series of JSON strings.
        Returns:
            pd.Series: A series of JSON strings converted to a common format.
    process_request(request_raw: DataFrame) -> DataFrame:
        Processes raw inference request DataFrame by transforming and unpacking JSON columns, extracting features and predictions, and adding metadata.
        Args:
            request_raw (DataFrame): The raw inference request DataFrame.
        Returns:
            DataFrame: The processed DataFrame with extracted features, predictions, and additional metadata.
"""

def read_inference_log_from_table(table):
    """
    Read inference logs from a table for the current day.
    """

    # Get the current date
    current_day = current_date()

    # Read the inference logs from the table and filter for the current day
    inference_df = spark.read.table(table) \
                        .where("status_code = 200") \
                        .where(col("date") == current_day)
    
    return inference_df

def convert_to_record_json(json_str: str):
    try:
        request = json.loads(json_str)
    except json.JSONDecodeError:
        return json_str  # Return original string if parsing fails
    
    output = []
    if isinstance(request, dict) and "inputs" in request:
        if isinstance(request["inputs"], list) and all(isinstance(i, list) for i in request["inputs"]):
            output.extend([{
                "request_index": idx,  # Track input order
                "input_0": values[0],
                "input_1": values[1],
                "input_2": values[2],
                "input_3": values[3]
            } for idx, values in enumerate(request["inputs"])])
    return json.dumps(output) if output else json_str

@F.pandas_udf(T.StringType())
def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
    return json_strs.apply(convert_to_record_json)

def process_request(request_raw: DataFrame):
    # Convert timestamp to human-readable format
    requests_timestamped = request_raw.withColumn(
        "timestamp_ms", from_unixtime(F.col("timestamp_ms") / 1000).cast(T.TimestampType())
    )

    # Extract and process JSON request column
    requests_unpacked = requests_timestamped.withColumn("request", json_consolidation_udf(F.col("request")))
    requests_unpacked = requests_unpacked.withColumn(
        "request",
        F.from_json(F.col("request"), T.ArrayType(T.StructType([
            T.StructField("request_index", T.IntegerType()),  # Preserve input order
            T.StructField("input_0", T.DoubleType()),
            T.StructField("input_1", T.DoubleType()),
            T.StructField("input_2", T.DoubleType()),
            T.StructField("input_3", T.DoubleType())
        ])))
    )

    # Correct aliasing for posexplode
    requests_unpacked = requests_unpacked.selectExpr("*", "posexplode(request) AS (pos, col)")

    # Extract feature columns
    for col_name in ["input_0", "input_1", "input_2", "input_3"]:
        requests_unpacked = requests_unpacked.withColumn(col_name, F.col(f"col.{col_name}"))

    # Drop unnecessary intermediate columns
    requests_unpacked = requests_unpacked.drop("col")

    # Parse response column safely
    requests_unpacked = requests_unpacked.withColumn(
        "response", F.from_json(F.col("response"), T.StructType([
            T.StructField("predictions", T.ArrayType(T.IntegerType()))
        ]))
    )

    # Explode predictions while maintaining correct order using posexplode
    requests_unpacked = requests_unpacked.selectExpr("*", "posexplode(response.predictions) AS (prediction_index, occupancy)")

    # Ensure predictions and inputs are correctly mapped
    requests_unpacked = requests_unpacked.filter(F.col("pos") == F.col("prediction_index"))

    # Drop unnecessary columns
    request_cleaned = requests_unpacked.drop("request", "response", "request_metadata", "pos", "prediction_index")

    # Add a placeholder model_id column
    final_df = request_cleaned.withColumn("model_id", F.lit(0).cast(T.IntegerType()))

    # Rename feature columns
    df = final_df.withColumnRenamed("input_0", "Temperature") \
                 .withColumnRenamed("input_1", "Light") \
                 .withColumnRenamed("input_2", "CO2") \
                 .withColumnRenamed("input_3", "HumidityRatio")

    # Remove duplicate rows
    df = df.dropDuplicates()

    return df