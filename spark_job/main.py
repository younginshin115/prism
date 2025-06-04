from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, struct, collect_list
from pyspark.sql.types import StructType, StringType, IntegerType, ArrayType, FloatType

from keras.models import load_model
from utils.prediction_utils import extract_model_metadata
from utils.minmax_utils import static_min_max
from data.live_process import process_packet_dicts
from data.parser import parse_labels_multiclass

model_path = "/app/models/10t-10n-IDS201X-LUCID-multi.h5"

# 모델 로드
model = load_model(model_path)

# 모델 메타데이터 추출
time_window, max_flow_len, model_name_string = extract_model_metadata(model_path)

# 정적 최소/최대 값 계산
mins, maxs = static_min_max(time_window)

# 라벨 맵 생성
label_map_dict, label_name_to_index = parse_labels_multiclass("DOS2017")

def debug_batch_factory(dataset_type, labels, max_flow_len, time_window):
    def debug_batch(df, batch_id):
        print(f"=== [Batch {batch_id}] ===")

        # packet_list 컬럼 안에서 struct 원소들을 꺼냄
        packet_dicts = df.selectExpr("explode(packet_list) as packet") \
                         .select("packet.*") \
                         .toLocalIterator()
        
        # Row 객체를 dict로 변환
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
            print("First flow sample:", samples[0])

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

    # ✅ 여기서 factory로 만든 함수 바인딩!
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
