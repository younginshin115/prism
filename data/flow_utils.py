# data/flow_utils.py
def dataset_to_list_of_fragments(dataset):
    """
    Convert flows into flat lists of fragments and labels.

    Args:
        dataset (List[Tuple[Tuple, Dict]]): Preprocessed flow dataset

    Returns:
        Tuple:
            - List of fragment arrays (X)
            - List of labels (y)
            - List of 5-tuples (keys)
            - List of metadata dicts (X_metadata)
    """
    keys = []
    X = []
    y = []
    X_metadata = []

    for flow_id, flow_data in dataset:
        label = flow_data.get('label', 0)
        src_ip = flow_id[0]
        dst_ip = flow_id[2]

        for key, fragment in flow_data.items():
            if isinstance(key, float):  # start_time_window
                X.append(fragment)
                y.append(label)
                keys.append(flow_id)

                # Try to fetch associated timestamp
                key_str = str(key)
                timestamp = flow_data.get(f"timestamp_{key_str}", None)
                if timestamp is None:
                    print(f"[Warning] Missing timestamp for flow_id={flow_id}, key={key}")
                X_metadata.append({
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "timestamp": timestamp
                })

    return X, y, keys, X_metadata


