from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, struct, collect_list
from pyspark.sql.types import StructType, StringType, IntegerType, ArrayType, FloatType

from keras.models import load_model
from utils.prediction_utils import extract_model_metadata, warm_up_model
from utils.minmax_utils import static_min_max
from data.live_process import process_packet_dicts
from data.parser import parse_labels_multiclass
from data.flow_utils import dataset_to_list_of_fragments
from core.prediction_runner import run_prediction_loop

# Load the trained Keras model
model_path = "/app/models/10t-10n-IDS201X-LUCID-multi.h5"
model = load_model(model_path)

# Extract model metadata (e.g., time window size, max flow length)
time_window, max_flow_len, model_name_string = extract_model_metadata(model_path)

# Load min/max values for feature normalization
mins, maxs = static_min_max(time_window)

# Parse label mapping for multiclass classification
label_map_dict, label_name_to_index = parse_labels_multiclass("DOS2017")

# Run warm-up inference to avoid cold start delays
warm_up_model(model)

def prediction_batch_handler_factory(dataset_type, labels, max_flow_len, time_window):
    """
    Create a function to process each micro-batch for prediction.
    """
    def prediction_batch_handler(df, batch_id):
        print(f"\n=== [Batch {batch_id}] ===")

        # Flatten the batch and convert to list of packet dictionaries
        packet_dicts = df.selectExpr("explode(packet_list) as packet") \
                        .select("packet.*") \
                        .toLocalIterator()

        packet_dicts = [row.asDict() for row in packet_dicts]

        # Convert raw packets into flow-based samples
        samples = process_packet_dicts(
            packet_dicts,
            dataset_type=dataset_type,
            in_labels=labels,
            max_flow_len=max_flow_len,
            time_window=time_window
        )

        print("Sample Count:", len(samples))

        # Run prediction only if samples exist
        if samples:
            X_raw, Y_true, _, X_metadata = dataset_to_list_of_fragments(samples)

            run_prediction_loop(
                X_raw=X_raw,
                Y_true=Y_true,
                X_metadata=X_metadata,
                model=model,
                model_name=model_name_string,
                source_name=f"batch_{batch_id}",
                mins=mins,
                maxs=maxs,
                max_flow_len=max_flow_len,
                label_mode="multi"
            )

    return prediction_batch_handler

def main():
    """
    Entry point: Configure Spark Streaming to read from Kafka and trigger prediction.
    """
    spark = SparkSession.builder \
        .appName("PRISM Kafka Packet Consumer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Read Kafka stream (JSON format)
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "packets") \
        .option("startingOffsets", "latest") \
        .load()

    # Define packet schema for parsing JSON messages
    packet_schema = StructType() \
        .add("timestamp", StringType()) \
        .add("src_ip", StringType()) \
        .add("dst_ip", StringType()) \
        .add("protocol", StringType()) \
        .add("length", IntegerType()) \
        .add("features", ArrayType(FloatType()))

    # Extract and parse JSON payload
    json_df = df.selectExpr("CAST(value AS STRING) as json_str")
    parsed_df = json_df.select(from_json(col("json_str"), packet_schema).alias("packet")).select("packet.*")
    parsed_df = parsed_df.withColumn("event_time", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))

    # Group packets into 5-second windows with watermarking for late data handling
    grouped_df = parsed_df.withWatermark("event_time", "1 minute").groupBy(
        window(col("event_time"), "5 seconds")
    ).agg(
        collect_list(
            struct("timestamp", "src_ip", "dst_ip", "protocol", "length", "features")
        ).alias("packet_list")
    )

    # Define the function to handle each micro-batch
    prediction_batch_handler = prediction_batch_handler_factory(
        dataset_type="DOS2017",
        labels=label_map_dict,
        max_flow_len=max_flow_len,
        time_window=time_window
    )

    # Start the streaming query and attach the prediction handler
    query = grouped_df.writeStream \
        .foreachBatch(prediction_batch_handler) \
        .outputMode("update") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
