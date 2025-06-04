# utils/prediction_logger.py

from elasticsearch import Elasticsearch

es = Elasticsearch("http://elasticsearch:9200")

def log_sample_predictions(
    X, Y_pred, Y_probs, timestamps, src_ips, dst_ips,
    model_name, source_name, prediction_time
):
    """
    Print per-sample prediction logs for debugging or future Elasticsearch indexing.

    Args:
        X (np.ndarray): Input feature vectors, shape (N, flow_len, feature_dim)
        Y_pred (np.ndarray): Predicted labels, shape (N,)
        Y_probs (np.ndarray): Predicted probabilities (sigmoid or softmax max), shape (N,)
        timestamps (list): List of timestamps per sample
        src_ips (list): List of source IPs per sample
        dst_ips (list): List of destination IPs per sample
        model_name (str): Model name
        source_name (str): Source identifier
        prediction_time (float): Time taken for prediction
    """
    for i in range(len(Y_pred)):
        doc = {
            "timestamp": timestamps[i],
            "src_ip": src_ips[i],
            "dst_ip": dst_ips[i],
            "packet_count": X[i].shape[0] if hasattr(X[i], 'shape') else len(X[i]),
            "label": int(Y_pred[i]),
            "probability": float(Y_probs[i]),
            "model": model_name,
            "source": source_name,
            "prediction_time": prediction_time
        }
        print("[Sample Prediction]", doc)

        try:
            es.index(index="packet-predictions", document=doc)
        except Exception as e:
            print("[Elasticsearch ERROR]", e)