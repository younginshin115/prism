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
            - List of dict metadata (e.g., timestamp, src_ip, dst_ip)
    """
    keys = []
    X = []
    y = []
    metadata = []

    for flow_id, flow_data in dataset:
        label = flow_data.get('label', 0)

        # 예시: flow_id = (timestamp, src_ip, dst_ip, proto)
        timestamp, src_ip, dst_ip, *_ = flow_id

        for key, fragment in flow_data.items():
            if key != 'label':
                X.append(fragment)
                y.append(label)
                keys.append(flow_id)
                metadata.append({
                    "timestamp": timestamp,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                })

    return X, y, keys, metadata

