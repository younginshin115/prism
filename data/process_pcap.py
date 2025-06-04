"""
process_pcap.py

Contains functions for processing offline or live traffic pcap files,
extracting per-packet features and assigning DDoS labels.
"""

import time
import numpy as np
from collections import OrderedDict
import pyshark

from data.parser import parse_packet


def process_pcap(pcap_file, dataset_type, in_labels, max_flow_len,
                 labelled_flows, max_flows=0, traffic_type='all', time_window=10):
    """
    Processes a pcap file and extracts labeled flows grouped by time window.

    Args:
        pcap_file (str): path to the .pcap file
        dataset_type (str): dataset identifier
        in_labels (dict): precomputed label dictionary
        max_flow_len (int): max packets per flow
        labelled_flows (list): output container for processed flows
        max_flows (int): optional limit on number of flows
        traffic_type (str): 'all', 'ddos', or 'benign'
        time_window (float): window size for flow grouping (seconds)
    """
    start_time = time.time()
    temp_dict = OrderedDict()
    start_time_window = -1

    pcap_name = pcap_file.split("/")[-1]
    print("Processing file:", pcap_name)

    cap = pyshark.FileCapture(pcap_file)
    for i, pkt in enumerate(cap):
        if i % 1000 == 0:
            print(pcap_name, "packet #", i)

        if start_time_window == -1 or float(pkt.sniff_timestamp) > start_time_window + time_window:
            start_time_window = float(pkt.sniff_timestamp)

        pf = parse_packet(pkt)
        store_packet(pf, temp_dict, start_time_window, max_flow_len)

        if max_flows > 0 and len(temp_dict) >= max_flows:
            break

    apply_labels(temp_dict, labelled_flows, in_labels, traffic_type)
    print('Completed file {} in {:.2f} seconds.'.format(pcap_name, time.time() - start_time))


def store_packet(pf, temp_dict, start_time_window, max_flow_len):
    """
    Adds a parsed packet to the correct flow in the temporary storage.

    Args:
        pf (PacketFeatures): parsed packet
        temp_dict (OrderedDict): flow-level storage
        start_time_window (float): base timestamp for grouping
        max_flow_len (int): max number of packets per flow
    """
    if pf is None:
        return temp_dict

    flow_id = None
    if pf.id_fwd in temp_dict:
        flow_id = pf.id_fwd
    elif pf.id_bwd in temp_dict:
        flow_id = pf.id_bwd
    else:
        flow_id = pf.id_fwd  # default to fwd

    # Initialize flow entry
    if flow_id not in temp_dict:
        temp_dict[flow_id] = {'label': 0}

    # Initialize time window entry
    if start_time_window not in temp_dict[flow_id]:
        temp_dict[flow_id][start_time_window] = np.array([pf.features_list])
        key_str = str(start_time_window)
        temp_dict[flow_id][f"timestamp_{key_str}"] = pf.timestamp 
    else:
        if temp_dict[flow_id][start_time_window].shape[0] < max_flow_len:
            temp_dict[flow_id][start_time_window] = np.vstack(
                [temp_dict[flow_id][start_time_window], pf.features_list]
            )

    return temp_dict

def apply_labels(flows, labelled_flows, labels, traffic_type):
    """
    Assign labels to each flow and store the result if it matches the type.

    Args:
        flows (OrderedDict): extracted flows
        labelled_flows (List): output container
        labels (dict): label mapping
        traffic_type (str): 'all', 'ddos', 'benign'
    """
    label_counter = {}

    for five_tuple, flow in flows.items():
        short_key = (five_tuple[0], five_tuple[2])  # src_ip, dst_ip
        label = labels.get(short_key, 0)
        flow['label'] = label

        # Count label usage
        label_counter[label] = label_counter.get(label, 0) + 1

        if traffic_type == 'ddos' and label == 0:
            continue
        elif traffic_type == 'benign' and label > 0:
            continue

        labelled_flows.append((five_tuple, flow))

    print("[Debug] Label distribution in current file:", label_counter)
