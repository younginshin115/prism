# utils/prediction_utils.py

import os
import glob
import csv
import time
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

from utils.constants import PREDICT_HEADER
from utils.path_utils import get_output_path
from data.data_loader import load_dataset


def load_model(model_path: str):
    """
    Load a Keras model from the given .h5 file path.

    Args:
        model_path (str): Full path to the .h5 model file

    Returns:
        keras.Model: Loaded Keras model
    """
    return keras_load_model(model_path)


def extract_filename_prefix(model_filename: str) -> str:
    """
    Extract prefix from model filename to match corresponding datasets.

    Example:
        '10t-10n-MyModel.h5' → '10t-10n-'

    Args:
        model_filename (str): Filename of the model

    Returns:
        str: Prefix to identify corresponding dataset files
    """
    parts = model_filename.split('-')
    return f"{parts[0].strip()}-{parts[1].strip()}-"


def warm_up_model(model, sample_file: str = None, input_shape: tuple = None):
    """
    Perform a warm-up forward pass to initialize GPU memory and avoid first-time latency.

    Args:
        model (keras.Model): Loaded Keras model
        sample_file (str, optional): Path to HDF5 file to load dummy input from
        input_shape (tuple, optional): Shape of a dummy input (e.g., (time_window, flow_len, 1))

    Raises:
        ValueError: If neither sample_file nor input_shape is provided
    """
    if sample_file:
        X, _ = load_dataset(sample_file)
        dummy_input = X[:1]
    elif input_shape:
        dummy_input = np.zeros((1,) + input_shape)  # Add batch dimension
    else:
        raise ValueError("Either sample_file or input_shape must be provided.")

    _ = model.predict(dummy_input, batch_size=1)


def get_dataset_files(dataset_folder: str) -> list:
    """
    Return a list of test dataset files (*.hdf5) in the given folder.

    Args:
        dataset_folder (str): Path to folder containing test datasets

    Returns:
        list: List of test file paths
    """
    return glob.glob(os.path.join(dataset_folder, "*test.hdf5"))


def extract_model_metadata(model_path: str):
    """
    Extract time_window, max_flow_len, and model name from the model filename.

    Args:
        model_path (str): Full path to model (.h5)

    Returns:
        tuple: (time_window: int, max_flow_len: int, model_name_string: str)
    """
    model_filename = os.path.basename(model_path)
    filename_prefix = extract_filename_prefix(model_filename)
    time_window = int(filename_prefix.split('t-')[0])
    max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
    model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
    return time_window, max_flow_len, model_name_string


def setup_prediction_output(output_folder: str) -> tuple:
    """
    Prepare CSV file and writer for saving prediction logs.

    Args:
        output_folder (str): Folder path where predictions will be saved

    Returns:
        tuple: (predict_file, csv_writer)
    """

    predict_file = open(
        get_output_path(output_folder, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
        'a',
        newline=''
    )
    predict_file.truncate(0)
    predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
    predict_writer.writeheader()
    predict_file.flush()
    return predict_file, predict_writer
