# data/parser.py

def parse_packet_dict(pkt: dict):
    """
    Convert a raw packet dictionary (from Kafka/Spark) into parsed format for flow processing.

    Args:
        pkt (dict): Dictionary with keys: timestamp, src_ip, dst_ip, protocol, length

    Returns:
        dict: Parsed packet format for flow aggregation
    """
    return {
        "timestamp": pkt.get("timestamp"),
        "src_ip": pkt.get("src_ip"),
        "dst_ip": pkt.get("dst_ip"),
        "protocol": pkt.get("protocol"),
        "length": pkt.get("length")
    }
