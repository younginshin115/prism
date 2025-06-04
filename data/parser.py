# data/parser.py
import hashlib
import ipaddress
import socket

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils.constants import PROTOCOLS, POWERS_OF_TWO

# Global protocol vectorizer
vector_proto = CountVectorizer()
vector_proto.fit_transform(PROTOCOLS).todense()


class PacketFeatures:
    """
    Represents the extracted features from a single network packet.

    Attributes:
        id_fwd (tuple): 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol)
        id_bwd (tuple): reversed 5-tuple (dst_ip, dst_port, src_ip, src_port, protocol)
        features_list (List[int | float]): extracted numerical features
    """
    def __init__(self):
        self.id_fwd = (0, 0, 0, 0, 0)
        self.id_bwd = (0, 0, 0, 0, 0)
        self.features_list = []

    def __str__(self):
        return f"{self.id_fwd} -> {self.features_list}"


def get_ddos_flows(attackers, victims):
    """
    Expand attackers and victims from IP or subnet strings into IP lists.

    Args:
        attackers (str): attacker IP or CIDR subnet (e.g., '192.168.0.0/24')
        victims (str): victim IP or CIDR subnet

    Returns:
        dict: dictionary with 'attackers' and 'victims' as lists of IP strings
    """
    ddos_flows = {}

    ddos_flows['attackers'] = (
        [str(ip) for ip in ipaddress.IPv4Network(attackers).hosts()]
        if '/' in attackers else [str(ipaddress.IPv4Address(attackers))]
    )
    ddos_flows['victims'] = (
        [str(ip) for ip in ipaddress.IPv4Network(victims).hosts()]
        if '/' in victims else [str(ipaddress.IPv4Address(victims))]
    )

    return ddos_flows


def parse_labels(dataset_type=None, attackers=None, victims=None, label=1):
    """
    Construct a label dictionary for attacker-victim flows.

    Args:
        dataset_type (str): predefined dataset name (e.g., 'DOS2018')
        attackers (str): optional attacker IP or subnet (overrides dataset_type)
        victims (str): optional victim IP or subnet (overrides dataset_type)
        label (int): label value to assign (default = 1)

    Returns:
        dict: mapping of 5-tuple flows (src, dst) to label
    """
    from data.ddos_specs import DDOS_ATTACK_SPECS

    output_dict = {}

    if attackers and victims:
        ddos_flows = get_ddos_flows(attackers, victims)
    elif dataset_type and dataset_type in DDOS_ATTACK_SPECS:
        ddos_flows = DDOS_ATTACK_SPECS[dataset_type]
    else:
        return None

    for attacker in ddos_flows['attackers']:
        for victim in ddos_flows['victims']:
            key_fwd = (attacker, victim)
            key_bwd = (victim, attacker)

            output_dict[key_fwd] = label
            output_dict[key_bwd] = label

    return output_dict


def parse_packet(pkt):
    """
    Extract a feature vector from a packet using pyshark.

    Args:
        pkt (pyshark.packet.packet.Packet): parsed packet object

    Returns:
        PacketFeatures: parsed feature vector object or None if packet is invalid
    """
    pf = PacketFeatures()
    tmp_id = [0] * 5

    try:
        pf.features_list.append(float(pkt.sniff_timestamp))  # timestamp
        pf.features_list.append(int(pkt.ip.len))             # packet length
        pf.features_list.append(
            int(hashlib.sha256(str(pkt.highest_layer).encode()).hexdigest(), 16) % 10**8
        )
        pf.features_list.append(int(int(pkt.ip.flags, 16)))  # IP flags

        tmp_id[0] = str(pkt.ip.src)
        tmp_id[2] = str(pkt.ip.dst)

        protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray()[0]
        protocols = [1 if i >= 1 else 0 for i in protocols]
        proto_value = int(np.dot(np.array(protocols), POWERS_OF_TWO))
        pf.features_list.append(proto_value)

        protocol = int(pkt.ip.proto)
        tmp_id[4] = protocol

        if pkt.transport_layer:
            if protocol == socket.IPPROTO_TCP:
                tmp_id[1] = int(pkt.tcp.srcport)
                tmp_id[3] = int(pkt.tcp.dstport)
                pf.features_list += [
                    int(pkt.tcp.len),
                    int(pkt.tcp.ack),
                    int(pkt.tcp.flags, 16),
                    int(pkt.tcp.window_size_value),
                    0, 0  # UDP, ICMP placeholders
                ]
            elif protocol == socket.IPPROTO_UDP:
                pf.features_list += [0, 0, 0, 0]
                tmp_id[1] = int(pkt.udp.srcport)
                tmp_id[3] = int(pkt.udp.dstport)
                pf.features_list.append(int(pkt.udp.length))
                pf.features_list.append(0)  # ICMP placeholder
        elif protocol == socket.IPPROTO_ICMP:
            pf.features_list += [0, 0, 0, 0, 0]
            pf.features_list.append(int(pkt.icmp.type))
        else:
            pf.features_list += [0, 0, 0, 0, 0, 0]
            tmp_id[4] = 0

        pf.id_fwd = tuple(tmp_id)
        pf.id_bwd = (tmp_id[2], tmp_id[3], tmp_id[0], tmp_id[1], tmp_id[4])

        return pf

    except AttributeError:
        # Packet does not contain required fields (e.g., not IPv4)
        return None

def infer_label_by_ip(src_ip, dst_ip, dataset='CIC-DDoS-2019'):
    """
    Infers the attack type label based on source and destination IP addresses.

    This function uses a predefined mapping of attacker and victim IP addresses
    for different datasets (e.g., CIC-IDS-2017/2018/2019) to determine whether a flow
    belongs to a known attack type or is benign.

    Args:
        src_ip (str): Source IP address of the packet or flow.
        dst_ip (str): Destination IP address of the packet or flow.
        dataset (str): Dataset name used to look up IP-based attack rules.
                       Default is 'CIC-DDoS-2019'.

    Returns:
        str: The attack type name if the IP pair matches a known attack,
             otherwise 'Benign'.
    """
    from data.ddos_specs import DDOS_ATTACK_SPECS_EXTENDED  # IP-based rules for multiple datasets

    # Return 'Benign' if the dataset has no known attack definitions
    if dataset not in DDOS_ATTACK_SPECS_EXTENDED:
        return 'Benign'

    # Iterate through all attack types defined for this dataset
    for attack_type, spec in DDOS_ATTACK_SPECS_EXTENDED[dataset].items():
        # Match on known attacker and victim IP combinations
        if src_ip in spec['attackers'] and dst_ip in spec['victims']:
            return attack_type

    # No match found: considered benign
    return 'Benign'

def parse_labels_multiclass(dataset_type):
    """
    Construct a multi-class label dictionary for attacker-victim flows.

    This function uses the extended DDoS attack specifications to generate
    a mapping from (src_ip, dst_ip) tuples to integer class labels.
    It also returns a mapping from class name to class index for reference.

    Args:
        dataset_type (str): Dataset name (e.g., 'CIC-DDoS-2019') used to look up
                            attacker-victim flow definitions.

    Returns:
        tuple:
            - dict: Mapping of (src_ip, dst_ip) → integer label index
            - dict: Mapping of attack type (str) → label index (int)
    """
    from data.ddos_specs import DDOS_ATTACK_SPECS_EXTENDED

    output_dict = {}   # Mapping from (attacker, victim) pair to label index
    label_map = {}     # Mapping from attack type string to unique label index
    label_counter = 1  # Start from 1; label 0 is typically reserved for benign traffic

    specs = DDOS_ATTACK_SPECS_EXTENDED.get(dataset_type, {})
    
    # Iterate over all defined attack types for the given dataset
    for attack_type, spec in specs.items():
        # Assign a new integer index to this attack type if not already assigned
        if attack_type not in label_map:
            label_map[attack_type] = label_counter
            label_counter += 1

        # For each attacker-victim IP pair, assign the corresponding label
        for attacker in spec['attackers']:
            for victim in spec['victims']:
                key_fwd = (attacker, victim)
                key_bwd = (victim, attacker)
                output_dict[key_fwd] = label_map[attack_type]
                output_dict[key_bwd] = label_map[attack_type]

    return output_dict, label_map

def parse_packet_dict(pkt):
    pf = PacketFeatures()

    # Spark Row 또는 dict 처리
    is_dict = isinstance(pkt, dict)

    pf.features_list = pkt["features"] if is_dict else pkt.features

    src_ip = pkt["src_ip"] if is_dict else pkt.src_ip
    dst_ip = pkt["dst_ip"] if is_dict else pkt.dst_ip
    protocol = pkt["protocol"] if is_dict else pkt.protocol

    pf.id_fwd = (src_ip, 0, dst_ip, 0, protocol)
    pf.id_bwd = (dst_ip, 0, src_ip, 0, protocol)

    return pf
