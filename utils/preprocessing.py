# utils/preprocessing.py
import numpy as np

def scale_linear_bycolumn(rawpoints, mins, maxs, high=1.0, low=0.0):
    """
    Apply feature-wise min-max scaling to a 2D array.
    """
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)


def normalize_and_padding(X, mins, maxs, max_flow_len, padding=True):
    """
    Normalize and optionally pad each sample in the dataset.

    Args:
        X (list of np.ndarray): List of samples (flows)
        mins (np.ndarray): Min values per feature
        maxs (np.ndarray): Max values per feature
        max_flow_len (int): Desired number of packets per sample
        padding (bool): Whether to apply zero-padding

    Returns:
        list of np.ndarray: Normalized and padded samples
    """
    norm_X = []
    for sample in X:
        if sample.shape[0] > max_flow_len:
            sample = sample[:max_flow_len, ...]

        packet_nr = sample.shape[0]
        norm_sample = scale_linear_bycolumn(sample, mins, maxs, high=1.0, low=0.0)
        np.nan_to_num(norm_sample, copy=False)

        if padding:
            norm_sample = np.pad(
                norm_sample,
                ((0, max_flow_len - packet_nr), (0, 0)),
                'constant',
                constant_values=(0, 0)
            )

        norm_X.append(norm_sample)

    return norm_X


def padding(X, max_flow_len):
    """
    Apply zero-padding to all samples to reach max_flow_len packets.

    Args:
        X (list of np.ndarray): List of flow samples
        max_flow_len (int): Desired number of packets per sample

    Returns:
        list of np.ndarray: Padded samples
    """
    padded_X = []
    for sample in X:
        flow_nr = sample.shape[0]
        padded_sample = np.pad(
            sample,
            ((0, max_flow_len - flow_nr), (0, 0)),
            'constant',
            constant_values=(0, 0)
        )
        padded_X.append(padded_sample)

    return padded_X
