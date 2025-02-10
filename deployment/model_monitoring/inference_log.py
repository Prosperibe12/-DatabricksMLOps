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

def read_inference_logs_from_table(table):
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
        return json_str
    
    output = []
    if isinstance(request, dict):
        # handle different JSON formats and convert to common format
        if "dataframe_records" in request:
            output.extend(request["dataframe_records"])
        elif "dataframe_split" in request:
            dataframe_split = request["dataframe_split"]
            output.extend([dict(zip(dataframe_split["columns"], values)) for values in dataframe_split["data"]])
        elif "instances" in request:
            output.extend(request["instances"])
        elif "inputs" in request:
            if isinstance(request["inputs"], list) and all(isinstance(i, list) for i in request["inputs"]):
                output.extend([dict(zip(["input_{}".format(i) for i in range(len(values))], values)) for values in request["inputs"]])
            else:
                output.extend([dict(zip(request["inputs"].keys(), values)) for values in zip(*request["inputs"].values())])
        elif "predictions" in request:
            output.extend([{'predictions': prediction} for prediction in request["predictions"]])
        return json.dumps(output)
    else:
        # if the format is unsupported, return the original JSON string
        return json_str
    
@F.pandas_udf(T.StringType())
def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
    return json_strs.apply(convert_to_record_json) 

def process_request(request_raw: DataFrame):
    # transform timestamp_ms to a readable format
    requests_timestamped = request_raw.withColumn("timestamp_ms", from_unixtime(F.col("timestamp_ms") / 1000).cast(TimestampType()))
    # unpack JSON from the 'request' column only since 'response' is already structured
    requests_unpacked = requests_timestamped.withColumn("request", json_consolidation_udf(F.col("request"))) \
                                            .withColumn("request", F.explode(F.from_json(F.col("request"), T.ArrayType(T.StructType([
                                                T.StructField("input_0", T.DoubleType()),
                                                T.StructField("input_1", T.DoubleType()),
                                                T.StructField("input_2", T.DoubleType()),
                                                T.StructField("input_3", T.DoubleType())
                                            ])))))
    # Extract feature columns as scalar values
    feature_columns = ["input_0", "input_1", "input_2", "input_3"]
    for col_name in feature_columns:
        requests_unpacked = requests_unpacked.withColumn(col_name, F.col(f"request.{col_name}"))
    
    # extract predictions from the 'response' column without using the from_json
    requests_unpacked = requests_unpacked.withColumn("response", F.from_json(F.col("response"), T.StructType([T.StructField("predictions", T.ArrayType(T.IntegerType()))])))
    requests_unpacked = requests_unpacked.withColumn("occupancy", F.explode(F.col("response.predictions")))
    # drop unecessary columns
    request_cleaned = requests_unpacked.drop("request", "response", "request_metadata")
    # Add a placeholder model_id column
    final_df = request_cleaned.withColumn("model_id", F.lit(0).cast(T.IntegerType()))
    # rename columns
    df = final_df.withColumnRenamed("input_0", "Temperature") \
                    .withColumnRenamed("input_1", "Light") \
                    .withColumnRenamed("input_2", "CO2") \
                    .withColumnRenamed("input_3", "HumidityRatio")
    # Remove duplicate rows
    df = df.dropDuplicates()
    return df