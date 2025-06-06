# PRISM: Packet-based Real-time Intrusion Surveillance Monitor

**PRISM**는 패킷 단위 네트워크 트래픽을 실시간으로 수집·분석하고,  
DDoS 등 다양한 공격을 다중 분류하여 시각화하는 스트리밍 기반 침입 감지 시스템입니다.

## 개요

- 실시간 트래픽을 감지하여 다중 분류된 결과를 Elasticsearch에 저장합니다.
- Kibana 대시보드를 통해 시각화 및 알람 기능을 제공합니다.
- 패킷 수집부터 예측, 시각화까지 전 과정을 자동화한 경량 추론 파이프라인입니다.

## 시스템 구성도

```
PyShark 수집기 (Producer)
↓

Kafka (Topic: packets)
↓

Spark Structured Streaming
↓

Elasticsearch (Index: ddos_predictions)
↓

Kibana (Dashboard 및 Alerts)
```

## 기술 스택

| 구성 요소     | 기술                       |
| ------------- | -------------------------- |
| 패킷 수집     | Python + pyshark           |
| 메시지 큐     | Kafka                      |
| 스트리밍 분석 | Spark Structured Streaming |
| 추론 모델     | TensorFlow/Keras           |
| 데이터 저장   | Elasticsearch              |
| 시각화        | Kibana                     |
| 배포 환경     | Docker + docker-compose    |

## 주요 기능

- NIC 또는 `.pcap` 파일에서 실시간 트래픽 수집
- 다중 패킷 기반 Flow 구성
- DDoS를 포함한 멀티 클래스 분류 (예: DoS, PortScan, Normal 등)
- 예측 결과를 JSON 형태로 Elasticsearch에 저장
- Kibana를 통해 실시간 시각화 및 알람 구성

## 예측 결과 예시

```json
{
  "timestamp": "2025-05-28T11:00:00Z",
  "src_ip": "192.168.1.1",
  "dst_ip": "18.218.83.150",
  "packet_count": 120,
  "label": "DDoS",
  "probability": 0.983
}
```

## 빠른 시작

1. 모델 준비
   학습한 .h5 모델 파일을 /models 디렉토리에 저장합니다.

2. Docker 환경 실행

```bash
docker-compose up --build
```

2️⃣ Conda 환경 준비 (최초 1회)

```bash
conda create -n prism-env python=3.10 -y
conda activate prism-env
cd producer
pip install -r requirements.txt
```

Wireshark가 설치되어 있지 않다면:

```bash
sudo apt install tshark -y
sudo usermod -aG wireshark $USER
newgrp wireshark  # 권한 적용
```

3️⃣ 패킷 수집기 실행

```bash
conda activate prism-env
cd producer
python packet_producer.py
```

실행 결과 예시:

```csharp
[INFO] Capturing packets on eth0...
[SENT] {"timestamp": "2025-06-03T12:30:00Z", "src_ip": "192.168.0.1", ...}
```

🧪 Kafka 메시지 확인 (선택)
Kafka에 메시지가 정상적으로 들어가는지 확인하려면, Kafka 컨테이너에 진입해서 consumer를 실행합니다:

```bash
docker exec -it kafka bash
```

그 다음 컨테이너 내부에서 직접 명령어를 실행합니다:

```bash
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic packets --from-beginning
```

🧪 Kafka 토픽 확인 (선택)

Kafka에 packets 토픽이 자동 생성되었는지 확인하려면:

```bash
docker exec -it kafka kafka-topics \
  --bootstrap-server localhost:9092 --list
```

정상적으로 작동하면 아래처럼 출력됩니다:

```nginx
packets
```

> 위 명령은 Kafka 컨테이너 내부에서 실행되는 CLI 명령어입니다.
> docker-compose로 실행 중인 Kafka 컨테이너 이름이 kafka일 경우만 그대로 사용 가능하며,
> 컨테이너 이름이 다르면 docker ps로 이름 확인 후 수정하세요.

3. Kibana 접속
   주소: http://localhost:5601
   인덱스: ddos_predictions\*

## 대시보드 예시

- 실시간 예측 로그 테이블
- 시간대별 공격 탐지 건수
- 공격 유형 분포 (원형 차트)
- GeoIP 기반 공격 위치 지도
- 예측 확률 변화 추이 (선형 차트)
- 알람: 특정 클래스/확률 이상일 때 경고 발생 가능

## 디렉토리 구조

```bash
prism/
├── producer/               # pyshark + Kafka Producer
├── spark_job/              # Kafka → Spark 추론 → Elasticsearch 저장
├── dashboards/             # Kibana ndjson export 파일
├── models/                 # 저장된 학습 모델 (.h5)
├── docker/                 # Dockerfile, compose 설정
└── README.md
```

## 라이선스 및 기여

- 본 프로젝트는 학술 목적의 연구 데모입니다.
- 내부 확장이나 실무 전환 시, Kafka Topic 병렬 처리 및 인증 구성 필요
