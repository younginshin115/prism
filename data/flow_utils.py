# data/flow_utils.py

import random
from collections import defaultdict

def count_flows(preprocessed_flows):
    """
    Count the number of flows and fragments per class.

    Args:
        preprocessed_flows (List[Tuple[5-tuple, Dict]]): Flow data with labels

    Returns:
        Tuple:
            - flow_count_per_class (Dict[int, int])
            - fragment_count_per_class (Dict[int, int])
    """
    flow_count_per_class = defaultdict(int)
    fragment_count_per_class = defaultdict(int)

    for _, flow in preprocessed_flows:
        label = flow.get('label', 0)
        fragment_count = len(flow) - 1  # exclude metadata

        flow_count_per_class[label] += 1
        fragment_count_per_class[label] += fragment_count

    return dict(flow_count_per_class), dict(fragment_count_per_class)

def balance_dataset_by_under_sampling(flows, total_fragments=float("inf")):
    """
    Perform class balancing by under-sampling the benign (label 0) class,
    keeping all attack flows. Optionally limit the total number of fragments.

    Args:
        flows (List[Tuple]): List of flows (5-tuple, dict)
        total_fragments (int): Max number of total fragments to keep (default: no limit)

    Returns:
        Tuple:
            - balanced_flows (List[Tuple])
            - fragment_counter (Dict[int, int]): Actual fragment count per class
    """
    # Separate benign and attack flows
    benign_flows = [f for f in flows if f[1].get("label", 0) == 0]
    attack_flows = [f for f in flows if f[1].get("label", 0) != 0]

    # Count total fragments from attack flows
    _, frag_count_per_class = count_flows(attack_flows)
    total_attack_fragments = sum(frag_count_per_class.values())

    # Determine target fragment count for benign
    target_benign_fragments = min(total_attack_fragments, total_fragments / 2)

    # Under-sample benign flows
    random.shuffle(benign_flows)
    selected_benign_flows = []
    benign_fragments = 0

    for flow in benign_flows:
        fragment_count = len(flow[1]) - 1
        if benign_fragments + fragment_count > target_benign_fragments:
            continue
        selected_benign_flows.append(flow)
        benign_fragments += fragment_count

    # Optionally trim attack fragments if total_fragments is also limiting them
    if total_fragments < float("inf"):
        selected_attack_flows = []
        attack_fragments = 0
        random.shuffle(attack_flows)

        for flow in attack_flows:
            fragment_count = len(flow[1]) - 1
            if attack_fragments + fragment_count > total_fragments / 2:
                continue
            selected_attack_flows.append(flow)
            attack_fragments += fragment_count
    else:
        selected_attack_flows = attack_flows

    # Combine and shuffle
    balanced_flows = selected_benign_flows + selected_attack_flows
    random.shuffle(balanced_flows)

    # Count the actual number of fragments per class
    _, balanced_fragment_counter = count_flows(balanced_flows)

    return balanced_flows, balanced_fragment_counter

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
    """
    keys = []
    X = []
    y = []

    for flow_id, flow_data in dataset:
        label = flow_data.get('label', 0)
        for key, fragment in flow_data.items():
            if key != 'label':
                X.append(fragment)
                y.append(label)
                keys.append(flow_id)

    return X, y, keys
