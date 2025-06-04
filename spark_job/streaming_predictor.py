# spark_job/streaming_predictor.py

from data.live_process import process_packet_dicts
from data.flow_utils import dataset_to_list_of_fragments
from core.prediction_runner import run_prediction_loop

def run_spark_batch(batch_df, batch_id, model, model_name_string, label_map, writer, max_flow_len, mins, maxs):
    """
    Process a micro-batch from Spark Structured Streaming, and run inference.

    Args:
        batch_df (DataFrame): Spark batch dataframe
        batch_id (int): Batch ID (automatically provided by Spark)
        model: Loaded model (.h5)
        label_map (dict): For apply_labels
        writer: CSV writer or elastic writer
        max_flow_len (int): Max flow length
        mins, maxs: For normalization
    """
    pandas_df = batch_df.select("packet_list").toPandas()

    for packet_list in pandas_df["packet_list"]:
        # 1. 전처리
        samples = process_packet_dicts(packet_list, dataset_type="DOS2017", in_labels=label_map, max_flow_len=max_flow_len)

        if not samples:
            continue

        # 2. Feature 변환
        X, Y_true, keys = dataset_to_list_of_fragments(samples)

        # 3. 추론 실행
        run_prediction_loop(
            X_raw=X,
            Y_true=Y_true,
            model=model,
            model_name=model_name_string,
            source_name="stream",
            mins=mins,
            maxs=maxs,
            max_flow_len=max_flow_len,
            writer=writer,
            label_mode="stream"
        )
