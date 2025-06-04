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

model_path = "/app/models/10t-10n-IDS201X-LUCID-multi.h5"

# 모델 로드 및 메타데이터 준비
model = load_model(model_path)
time_window, max_flow_len, model_name_string = extract_model_metadata(model_path)
mins, maxs = static_min_max(time_window)
label_map_dict, label_name_to_index = parse_labels_multiclass("DOS2017")
warm_up_model(model)


def debug_batch_factory(dataset_type, labels, max_flow_len, time_window):
    def debug_batch(df, batch_id):
        print(f"\n=== [Batch {batch_id}] ===")

        packet_dicts = df.selectExpr("explode(packet_list) as packet") \
                        .select("packet.*") \
                        .toLocalIterator()

        packet_dicts = [row.asDict() for row in packet_dicts]

        samples = process_packet_dicts(
            packet_dicts,
            dataset_type=dataset_type,
            in_labels=labels,
            max_flow_len=max_flow_len,
            time_window=time_window
        )

        print("Sample Count:", len(samples))
        if samples:
            X, Y_true, keys = dataset_to_list_of_fragments(samples)
            run_prediction_loop(
                X_raw=X,
                Y_true=Y_true,
                model=model,
                model_name=model_name_string,
                source_name=f"batch_{batch_id}",
                mins=mins,
                maxs=maxs,
                max_flow_len=max_flow_len,
                writer=None,  # CSV로 저장하지 않음
                label_mode="multi"
            )

    return debug_batch


def main():
    spark = SparkSession.builder \
        .appName("PRISM Kafka Packet Consumer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "packets") \
        .option("startingOffsets", "latest") \
        .load()

    packet_schema = StructType() \
        .add("timestamp", StringType()) \
        .add("src_ip", StringType()) \
        .add("dst_ip", StringType()) \
        .add("protocol", StringType()) \
        .add("length", IntegerType()) \
        .add("features", ArrayType(FloatType()))

    json_df = df.selectExpr("CAST(value AS STRING) as json_str")
    parsed_df = json_df.select(from_json(col("json_str"), packet_schema).alias("packet")).select("packet.*")
    parsed_df = parsed_df.withColumn("event_time", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))

    grouped_df = parsed_df.withWatermark("event_time", "1 minute").groupBy(
        window(col("event_time"), "5 seconds")
    ).agg(
        collect_list(
            struct("timestamp", "src_ip", "dst_ip", "protocol", "length", "features")
        ).alias("packet_list")
    )

    # Debug용 예측 함수 설정
    debug_batch = debug_batch_factory(
        dataset_type="DOS2017",
        labels=label_map_dict,
        max_flow_len=max_flow_len,
        time_window=time_window
    )

    query = grouped_df.writeStream \
        .foreachBatch(debug_batch) \
        .outputMode("update") \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
