from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, struct, collect_list
from pyspark.sql.types import StructType, StringType, IntegerType, ArrayType, FloatType


# spark_main.py (또는 별도 파일에)

from data.live_process import process_packet_dicts
from data.parser import parse_labels_multiclass

label_map_dict, label_name_to_index = parse_labels_multiclass("DOS2017")

def debug_batch(df, batch_id):
    pandas_df = df.select("packet_list").toPandas()

    for i, packet_list in enumerate(pandas_df["packet_list"]):
        print(f"\n=== [Batch {batch_id}] Group {i} ===")
        print(f"Packet Count: {len(packet_list)}")

        # 1. 전처리 실행
        samples = process_packet_dicts(
            packet_dicts=packet_list,
            dataset_type="DOS2017",
            in_labels=label_map_dict,     # 이건 전역에서 parse_labels로 불러와야 함
            max_flow_len=20
        )

        # 2. 전처리 결과 확인
        print(f"Generated {len(samples)} flow samples")
        if samples:
            print("First sample flow_id:", samples[0]["flow_id"])
            print("First sample label:", samples[0]["label"])
            print("First sample feature shape:", len(samples[0]["features"]), "x", len(samples[0]["features"][0]))



def main():
    spark = SparkSession.builder \
        .appName("PRISM Kafka Packet Consumer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 1. Kafka에서 메시지 수신
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "packets") \
        .option("startingOffsets", "latest") \
        .load()

    # 2. Kafka 메시지를 JSON으로 파싱
    packet_schema = StructType() \
        .add("timestamp", StringType()) \
        .add("src_ip", StringType()) \
        .add("dst_ip", StringType()) \
        .add("protocol", StringType()) \
        .add("length", IntegerType()) \
        .add("features", ArrayType(FloatType()))

    json_df = df.selectExpr("CAST(value AS STRING) as json_str")
    parsed_df = json_df.select(from_json(col("json_str"), packet_schema).alias("packet")).select("packet.*")

    # 3. event_time 컬럼 추가
    parsed_df = parsed_df.withColumn("event_time", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))

    # 4. window로 묶어서 packet_list 생성
    grouped_df = parsed_df.withWatermark("event_time", "1 minute").groupBy(
        window(col("event_time"), "5 seconds")
    ).agg(
        collect_list(
            struct("timestamp", "src_ip", "dst_ip", "protocol", "length", "features")
        ).alias("packet_list")
    )

    # 5. 전처리 테스트만 하고 싶으면 아래처럼 packet_list만 출력해보자!
    query = grouped_df.writeStream \
        .foreachBatch(debug_batch) \
        .outputMode("update") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
