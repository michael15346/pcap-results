import sys
import math
from pathlib import Path
from utils import lit2i
from const import PCAP_FILE_HEADER_LEN, PCAP_PACKET_HEADER_LEN
from encoder import Decoder, PCAP, IPv4, IPv6, TCP, UDP
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torch import tensor
import torch
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import cProfile



ZSTD_LEVEL = 1
BATCH_SIZE = 2048


def split_128bit(i):
    return [(i // (2 ** 32)) % 2 ** 32,
            i % 2 ** 32
            ]


def split_48bit(i):
    return [i // (2 ** 24),
            i % (2 ** 24)]


def to_dataset(packet: PCAP):
    #data = split_128bit(packet.timestamp)
    data = []
    data.extend([
            packet.len_original
            ])
    eth_payload = packet.payload
    ip_payload = eth_payload.payload
    if isinstance(ip_payload, IPv4):
        data.extend([
            ip_payload.id_,
            ip_payload.total_length,
            ip_payload.fragment_offset,
            ip_payload.src,
            ip_payload.dst,
            ])
    else:
        return None
    transport_payload = ip_payload.payload
    if isinstance(transport_payload, TCP):
        tcp_options = transport_payload.options
        data.extend([
            transport_payload.src,
            transport_payload.dst,
            transport_payload.seq_no,
            transport_payload.ack_no,
            transport_payload.data_off,
            transport_payload.window
            ])
    else:
        return None
    return data

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, seq_length):
        self.time_series = time_series
        self.seq_length = seq_length

    def __len__(self):
        return len(self.time_series) - self.seq_length - 1

    def __getitem__(self, index):

        ip_src_1 = self.time_series[index][4]
        ip_dst_1 = self.time_series[index][5]
        tcp_src_1 = self.time_series[index][6]
        tcp_dst_1 = self.time_series[index][7]
        key_1 = (ip_src_1, tcp_src_1, ip_dst_1, tcp_dst_1)

        ip_src_2 = self.time_series[index + 1][4]
        ip_dst_2 = self.time_series[index + 1][5]
        tcp_src_2 = self.time_series[index + 1][6]
        tcp_dst_2 = self.time_series[index + 1][7]
        key_2 = (ip_src_2, tcp_src_2, ip_dst_2, tcp_dst_2)
        while key_1 != key_2:
          
          index += 1
          if index == self.__len__():
            index = 0
          ip_src_1 = self.time_series[index][4]
          ip_dst_1 = self.time_series[index][5]
          tcp_src_1 = self.time_series[index][6]
          tcp_dst_1 = self.time_series[index][7]
          key_1 = (ip_src_1, tcp_src_1, ip_dst_1, tcp_dst_1)

          ip_src_2 = self.time_series[index + 1][4]
          ip_dst_2 = self.time_series[index + 1][5]
          tcp_src_2 = self.time_series[index + 1][6]
          tcp_dst_2 = self.time_series[index + 1][7]
          key_2 = (ip_src_2, tcp_src_2, ip_dst_2, tcp_dst_2)
        X = self.time_series[index, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        y = self.time_series[index + 1, [1]]
        return X, y


def split_into_sessions(d: np.ndarray):
    sess = dict()
    for i in range(d.shape[0]):
        ip_src = d[i][4]
        ip_dst = d[i][5]
        tcp_src = d[i][6]
        tcp_dst = d[i][7]
        key = (ip_src, tcp_src, ip_dst, tcp_dst)
        if key not in sess:
            sess[key] = [d[i]]
        else:
            sess[key].append(d[i])
    return np.array([x for xs in sess.values() for x in xs])


def main(argv):
    if len(argv) < 2:
        print(f"Usage: {argv[0]} PCAP_FILE")

    fpath_in = Path(argv[1])
    with open(fpath_in, 'rb') as f:
        data = f.read()
        n_bytes = len(data)

    file_header = data[:PCAP_FILE_HEADER_LEN]

    i = PCAP_FILE_HEADER_LEN
    n_packets = 0
    res = [file_header]
    encoder = Decoder()

    while i < n_bytes:
        n_packets += 1
        cpl = lit2i(data[i+8:i+12])

        p = encoder.process(data[i:i + PCAP_PACKET_HEADER_LEN + cpl])
        res.append(p)

        i += PCAP_PACKET_HEADER_LEN + cpl

    dataset = []
    for i in res:
        # print(i)
        if isinstance(i, PCAP):
            point = to_dataset(i)
            if point is not None:
                dataset.append(point)
    # print(dataset)
    print("============================")

    dataset = np.array(dataset, dtype='longdouble')
    dataset = split_into_sessions(dataset)
    #pd.DataFrame(dataset).to_csv('drive/MyDrive/sess_ord.csv')
    #print(dataset.shape)
    #means = dataset.mean(axis=0)
    #dataset -= means
    #std = dataset.std(axis=0)
    #std[std == 0] = 1
    #dataset /= std
    dataset = dataset.astype(np.float64)
    seq_length = 1
    datasets = random_split(TimeSeriesDataset(dataset, seq_length), (0.8, 0.2))
    d_x = [i[0] for i in datasets[0]]
    d_y = [i[1] for i in datasets[0]]
    #print(d_x)
    d_test_x = [i[0] for i in datasets[1]]
    d_test_y = [i[1] for i in datasets[1]]
    r = linear_model.LinearRegression()

    r.fit(d_x, d_y)
    print(r.coef_)
    pred = r.predict(d_test_x)
    head = [
            'len_original',
            'ip_id',
            'total_length',
            'fragment_offset',
            'ip_src',
            'ip_dst',
            'tcp_src',
            'tcp_dst',
            'tcp_seq_no',
            'tcp_ack_no',
            'tcp_data_off',
            'tcp_window'
            ]
    pd.DataFrame(pred, columns=['ip_id']).to_csv('pred.csv')
    pd.DataFrame(d_test_y, columns=['ip_id']).to_csv('target.csv')
    pd.DataFrame(d_test_x, columns=head).to_csv('inputs.csv')




if __name__ == "__main__":
    main(sys.argv)


