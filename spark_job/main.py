from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("PRISM Kafka Consumer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "packets") \
        .option("startingOffsets", "latest") \
        .load()

    df.selectExpr("CAST(value AS STRING)").writeStream \
        .format("console") \
        .option("truncate", "false") \
        .start() \
        .awaitTermination()

if __name__ == "__main__":
    main()
