import sys
import math
from pathlib import Path
from utils import lit2i
from const import PCAP_FILE_HEADER_LEN, PCAP_PACKET_HEADER_LEN
from encoder import Decoder, PCAP, IPv4, IPv6, TCP, UDP
import numpy as np
import pandas as pd
from torch.nn import Sequential, Linear, LSTM, ReLU
from torch.optim import SGD
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torch
from torch import nn
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
            transport_payload.window,
            ])
    else:
        return None
    return data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class UberModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.model = nn.Linear(8, 8)

    def forward(self, x):
        x = self.model(x)
        return x



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
        if ip_src_1 < ip_dst_1:
            key_1 = (ip_src_1, tcp_src_1, ip_dst_1, tcp_dst_1)
        else:
            key_1 = (ip_dst_1, tcp_dst_1, ip_src_1, tcp_src_1)

        ip_src_2 = self.time_series[index + 1][4]
        ip_dst_2 = self.time_series[index + 1][5]
        tcp_src_2 = self.time_series[index + 1][6]
        tcp_dst_2 = self.time_series[index + 1][7]
        if ip_src_2 < ip_dst_2:
            key_2 = (ip_src_2, tcp_src_2, ip_dst_2, tcp_dst_2)
        else:
            key_2 = (ip_dst_2, tcp_dst_2, ip_src_2, tcp_src_2)
        while key_1 != key_2:
          
          index += 1
          if index == self.__len__():
            index = 0
          ip_src_1 = self.time_series[index][4]
          ip_dst_1 = self.time_series[index][5]
          tcp_src_1 = self.time_series[index][6]
          tcp_dst_1 = self.time_series[index][7]
          if ip_src_1 < ip_dst_1:
              key_1 = (ip_src_1, tcp_src_1, ip_dst_1, tcp_dst_1)
          else:
              key_1 = (ip_dst_1, tcp_dst_1, ip_src_1, tcp_src_1)

          ip_src_2 = self.time_series[index + 1][4]
          ip_dst_2 = self.time_series[index + 1][5]
          tcp_src_2 = self.time_series[index + 1][6]
          tcp_dst_2 = self.time_series[index + 1][7]
          if ip_src_2 < ip_dst_2:
              key_2 = (ip_src_2, tcp_src_2, ip_dst_2, tcp_dst_2)
          else:
              key_2 = (ip_dst_2, tcp_dst_2, ip_src_2, tcp_src_2)
        X = self.time_series[index:index + self.seq_length, [0, 1, 2, 3, 8, 9, 10, 11]]
        y = self.time_series[index + 1 : index + 1 + self.seq_length, [0, 1, 2, 3, 8, 9, 10, 11]]
        return torch.tensor(X, dtype=torch.float32), \
            torch.tensor(y, dtype=torch.float32)



def eval_model(dataset, model, std, means):
    model.eval()
    test_loss = 0
    first = False
    with torch.no_grad():
        for batch, data in enumerate(dataset):
            inputs, targets = data[0].to("cuda"), data[1].to("cuda")
            preds = model(inputs)
            if not first:
                first = True
                print("inputs")
                head = [
                        'len_original',
                        'ip_id',
                        'total_length',
                        'fragment_offset',
                        'tcp_seq_no',
                        'tcp_ack_no',
                        'tcp_data_off',
                        'tcp_window'
                        ]
                #print(std.shape)
                inputs_np = inputs.cpu().numpy()# * std[[0, 1, 2, 3, 8, 9, 10, 11]] + means[[0, 1, 2, 3, 8, 9, 10, 11]]
                inputs_df = pd.DataFrame(inputs_np[:, 0, :], columns=head)
                inputs_df.to_csv('inputs.csv')
                print(inputs)
                print("targets")
                targets_np = targets.cpu().numpy()# * std[[0, 1, 2, 3, 8, 9, 10, 11]] + means[[0, 1, 2, 3, 8, 9, 10, 11]]
                targets_df = pd.DataFrame(targets_np[:, 0, :], columns=head)
                targets_df.to_csv('targets.csv')
                print(targets)
                print("preds")
                preds_np = preds.cpu().numpy()# * std[[0, 1, 2, 3, 8, 9, 10, 11]] + means[[0, 1, 2, 3, 8, 9, 10, 11]]
                preds_df = pd.DataFrame(preds_np[:, 0, :], columns=head)
                preds_df.to_csv('preds.csv')
                print(preds)
                
            #print(preds.shape)
            #print(targets.shape)
            # print(preds)
            # input()
            loss = nn.MSELoss()(preds, targets)
            test_loss += loss.item()
    return test_loss

def split_into_sessions(d: np.ndarray):
    sess = dict()
    for i in range(d.shape[0]):
        ip_src = d[i][4]
        ip_dst = d[i][5]
        tcp_src = d[i][6]
        tcp_dst = d[i][7]
        if ip_src < ip_dst:
            key = (ip_src, tcp_src, ip_dst, tcp_dst)
        else:
            key = (ip_dst, tcp_dst, ip_src, tcp_src)
        if key not in sess:
            sess[key] = [d[i]]
        else:
            sess[key].append(d[i])
    return np.array([x for xs in sess.values() for x in xs])


def main(argv):
    if len(argv) < 3:
        print(f"Usage: {argv[0]} PCAP_FILE MODEL_PT")

    fpath_in = Path(argv[1])
    with open(fpath_in, 'rb') as f:
        data = f.read()
        n_bytes = len(data)
    torch.serialization.add_safe_globals([UberModel])
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
    #means = dataset.mean(axis=0)
    #dataset -= means
    #std = dataset.std(axis=0)
    #std[std == 0] = 1
    #dataset /= std
    dataset = dataset.astype(np.float64)
    seq_length = 1
    d = TimeSeriesDataset(dataset, seq_length)
    l = len(d)
    datasets = (torch.utils.data.Subset(d, range(int(0.8 * l))), torch.utils.data.Subset(d, range(int(0.8 * l), int(0.9 * l))), torch.utils.data.Subset(d, range(int(0.9 * l), l)))
    #datasets = random_split(TimeSeriesDataset(dataset, seq_length), (0.8, 0.1, 0.1))
    d_train = datasets[0]
    d_test = datasets[1]
    d_val = datasets[2]
    dld = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=False)
    dld_test = DataLoader(d_test, batch_size=BATCH_SIZE, shuffle=False)
    dld_val = DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=False)


    with open(argv[2], 'rb') as f:
        model = torch.load(f, weights_only=False)

    eval_model(dld_val, model, None, None)





if __name__ == "__main__":
    main(sys.argv)

