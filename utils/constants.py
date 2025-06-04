# utils/constants.py
import numpy as np

SEED = 1
MAX_FLOW_LEN = 100 # number of packets
TIME_WINDOW = 10
TRAIN_SIZE = 0.90 # size of the training set wrt the total number of samples

PATIENCE = 10
DEFAULT_EPOCHS = 1000
VAL_HEADER = [
    'Model', 'Samples', 'Accuracy', 'F1Score', 'Precision', 'Recall',
    'Hyper-parameters', 'Validation Set', 'Label Mode'
]
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
PROTOCOLS = [
    'arp', 'data', 'dns', 'ftp', 'http', 'icmp',
    'ip', 'ssdp', 'ssl', 'telnet', 'tcp', 'udp'
]
POWERS_OF_TWO = np.array([2 ** i for i in range(len(PROTOCOLS))])