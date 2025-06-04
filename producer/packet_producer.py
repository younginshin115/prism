# producer/packet_producer.py

import pyshark
import json
from kafka import KafkaProducer

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
                payload = {
                    "timestamp": str(packet.sniff_time),
                    "src_ip": packet.ip.src if hasattr(packet, 'ip') else None,
                    "dst_ip": packet.ip.dst if hasattr(packet, 'ip') else None,
                    "protocol": packet.transport_layer if hasattr(packet, 'transport_layer') else "Unknown",
                    "length": int(packet.length)
                }
                producer.send(topic, payload)
                print(f"[KAFKA SENT] {payload}")
        except Exception as e:
            print(f"[ERROR] Packet skipped: {e}")

    print("[INFO] Stream ended.")

if __name__ == '__main__':
    run_packet_stream_to_kafka(source='eth0', topic='packets')