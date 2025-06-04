from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, IntegerType

def main():
    spark = SparkSession.builder \
        .appName("PRISM Kafka Packet Consumer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Kafka에서 메시지 수신
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "packets") \
        .option("startingOffsets", "latest") \
        .load()

    # Kafka에서 받은 메시지를 String으로 변환
    json_df = df.selectExpr("CAST(value AS STRING) as json_str")

    # JSON 스키마 정의
    packet_schema = StructType() \
        .add("timestamp", StringType()) \
        .add("src_ip", StringType()) \
        .add("dst_ip", StringType()) \
        .add("protocol", StringType()) \
        .add("length", IntegerType())

    # JSON 파싱
    parsed_df = json_df.select(from_json(col("json_str"), packet_schema).alias("packet")).select("packet.*")

    # 일단 콘솔로 출력
    query = parsed_df.writeStream \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
