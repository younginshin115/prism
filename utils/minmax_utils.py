# utils/minmax_utils.py

import numpy as np
from utils.constants import PROTOCOLS

# Feature list used for static min/max estimation
FEATURE_LIST = {
    'timestamp': [0, 10],
    'packet_length': [0, 1 << 16],
    'highest_layer': [0, 1 << 32],
    'IP_flags': [0, 1 << 16],
    'protocols': [0, 1 << len(PROTOCOLS)],
    'TCP_length': [0, 1 << 16],
    'TCP_ack': [0, 1 << 32],
    'TCP_flags': [0, 1 << 16],
    'TCP_window_size': [0, 1 << 16],
    'UDP_length': [0, 1 << 16],
    'ICMP_type': [0, 1 << 8]
}

def static_min_max(time_window=10):
    """
    Create min and max arrays based on predefined feature value ranges.
    """
    FEATURE_LIST['timestamp'][1] = time_window

    min_array = np.zeros(len(FEATURE_LIST))
    max_array = np.zeros(len(FEATURE_LIST))

    for i, (feature, value_range) in enumerate(FEATURE_LIST.items()):
        min_array[i] = value_range[0]
        max_array[i] = value_range[1]

    return min_array, max_array


def find_min_max(X, time_window=10):
    """
    Dynamically compute min and max feature values from a dataset.

    Args:
        X (list of np.ndarray): List of flow samples

    Returns:
        Tuple[np.ndarray, np.ndarray]: min and max arrays
    """
    sample_len = X[0].shape[1]
    max_array = np.zeros((1, sample_len))
    min_array = np.full((1, sample_len), np.inf)

    for feature in X:
        max_array = np.maximum(max_array, feature)
        min_array = np.minimum(min_array, feature)

    # Timestamp normalization
    max_array[0][0] = time_window
    min_array[0][0] = 0

    return min_array, max_array
