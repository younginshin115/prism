import numpy as np
import h5py
import glob

def load_dataset(path):
    """
    Load dataset from HDF5 file containing 'set_x' and 'set_y' datasets.

    Args:
        path (str): Glob pattern or filepath to HDF5 file

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y datasets
    """
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])
    set_y_orig = np.array(dataset["set_y"][:])

    X = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y = set_y_orig
    return X, Y


def count_packets_in_dataset(X_list):
    """
    Count number of non-zero packets across all samples.

    Args:
        X_list (list of np.ndarray): List of 4D tensors

    Returns:
        list[int]: Packet counts per sample set
    """
    packet_counters = []
    for X in X_list:
        TOT = X.sum(axis=2)
        packet_counters.append(np.count_nonzero(TOT))
    return packet_counters


def all_same(items):
    """
    Check if all elements in a list are equal.

    Args:
        items (list): Any list

    Returns:
        bool: True if all values are the same
    """
    return all(x == items[0] for x in items)
