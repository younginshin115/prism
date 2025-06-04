import pprint
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def report_results(
    Y_true,
    Y_pred,
    packets,
    model_name,
    data_source,
    prediction_time,
    writer,
    label_mode="binary"
):
    """
    Compute evaluation metrics and write prediction results to stdout and CSV.

    For binary classification, includes TPR/FPR/TNR/FNR metrics.
    For multi-class classification, calculates weighted F1-score and omits binary-specific metrics.

    Args:
        Y_true (np.ndarray or None): Ground truth labels (1D or one-hot)
        Y_pred (np.ndarray): Predicted labels (1D)
        packets (int): Number of packets observed in input
        model_name (str): Name of the model used for prediction
        data_source (str): Source name (e.g., filename or interface)
        prediction_time (float): Inference time in seconds
        writer (csv.DictWriter): Writer object for output CSV
        label_mode (str): "binary" or "multi"
    """
    # For binary classification, compute DDoS rate
    ddos_rate = '{:04.3f}'.format(np.sum(Y_pred) / Y_pred.shape[0]) if label_mode == "binary" else "N/A"

    if Y_true is not None and len(Y_true.shape) > 0:
        # Flatten inputs
        Y_true = Y_true.reshape(-1)
        Y_pred = Y_pred.reshape(-1)

        # Basic metrics
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, average="binary" if label_mode == "binary" else "weighted")

        if label_mode == "binary":
            # Extract confusion matrix entries
            tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()

            # Compute binary classification metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            # Not applicable for multi-class
            tpr = fpr = tnr = fnr = "N/A"

        row = {
            'Model': model_name,
            'Time': '{:04.3f}'.format(prediction_time),
            'Packets': packets,
            'Samples': Y_pred.shape[0],
            'DDOS%': ddos_rate,
            'Accuracy': '{:05.4f}'.format(accuracy),
            'F1Score': '{:05.4f}'.format(f1),
            'TPR': '{:05.4f}'.format(tpr) if tpr != "N/A" else tpr,
            'FPR': '{:05.4f}'.format(fpr) if fpr != "N/A" else fpr,
            'TNR': '{:05.4f}'.format(tnr) if tnr != "N/A" else tnr,
            'FNR': '{:05.4f}'.format(fnr) if fnr != "N/A" else fnr,
            'Source': data_source
        }

    else:
        # Fallback if Y_true is not provided
        row = {
            'Model': model_name,
            'Time': '{:04.3f}'.format(prediction_time),
            'Packets': packets,
            'Samples': Y_pred.shape[0],
            'DDOS%': ddos_rate,
            'Accuracy': "N/A",
            'F1Score': "N/A",
            'TPR': "N/A",
            'FPR': "N/A",
            'TNR': "N/A",
            'FNR': "N/A",
            'Source': data_source
        }

    # Output to console and CSV
    pprint.pprint(row, sort_dicts=False)
    if writer is not None:
        writer.writerow(row)
