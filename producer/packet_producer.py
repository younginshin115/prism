# producer/packet_producer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pyshark
import json
from kafka import KafkaProducer
from data.parser import parse_packet

def create_kafka_producer(bootstrap_servers='localhost:9092'):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def run_packet_stream_to_kafka(source: str, topic: str = 'packets', interface_mode=True):
    """
    Stream packets from NIC or pcap and send to Kafka.

    Args:
        source (str): NIC name (e.g. eth0) or pcap file path
        topic (str): Kafka topic to send packets
        interface_mode (bool): True if NIC interface, False if pcap file
    """
    producer = create_kafka_producer()
    print(f"[INFO] Starting packet stream from {source} to Kafka topic '{topic}'")

    capture = (
        pyshark.LiveCapture(interface=source)
        if interface_mode
        else pyshark.FileCapture(source)
    )

    for packet in capture.sniff_continuously():
        try:
            if hasattr(packet, 'ip'):
                pf = parse_packet(packet)
                if pf is None:
                    continue  # skip non-IPv4 or malformed packets

                payload = {
                    "timestamp": str(packet.sniff_time),
                    "src_ip": pf.id_fwd[0],
                    "dst_ip": pf.id_fwd[2],
                    "protocol": pf.id_fwd[4],
                    "length": int(packet.length),
                    "features": pf.features_list
                }

                producer.send(topic, payload)
                print(f"[KAFKA SENT] {payload}")
        except Exception as e:
            print(f"[ERROR] Packet skipped: {e}")

    print("[INFO] Stream ended.")

if __name__ == '__main__':
    run_packet_stream_to_kafka(source='eth0', topic='packets')