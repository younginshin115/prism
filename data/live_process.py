# data/live_process.py

from collections import OrderedDict
from data.parser import parse_packet_dict
from data.process_pcap import store_packet, apply_labels
from utils.constants import TIME_WINDOW
import time

def process_packet_dicts(packet_dicts, dataset_type, in_labels, max_flow_len, time_window=TIME_WINDOW, traffic_type='all'):
    """
    Process list of packet dicts into flows for inference (used in Spark or Kafka pipelines).

    Args:
        packet_dicts (List[dict]): Kafka/Spark 수신 패킷 리스트
        dataset_type (str): Dataset type (for label parsing)
        in_labels (dict): Flow label mappings
        max_flow_len (int): Max packets per flow
        time_window (int): Flow grouping window (in seconds)
        traffic_type (str): all / benign / ddos

    Returns:
        List of labeled flows
    """
    if not packet_dicts:
        return []

    start_time_window = time.time()
    temp_dict = OrderedDict()
    labelled_flows = []

    for pkt in packet_dicts:
        pf = parse_packet_dict(pkt)
        temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)

    apply_labels(temp_dict, labelled_flows, in_labels, traffic_type)
    return labelled_flows
