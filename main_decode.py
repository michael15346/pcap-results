import sys
from pathlib import Path
from utils import lit2i
from const import PCAP_FILE_HEADER_LEN, PCAP_PACKET_HEADER_LEN
from encoder import Decoder, PCAP, IPv4, TCP
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from sklearn import linear_model


def to_dataset(packet: PCAP):
    data = []
    data.extend([
            packet.timestamp,
            packet.len_original
            ])
    eth_payload = packet.payload
    ip_payload = eth_payload.payload
    if isinstance(ip_payload, IPv4):
        data.extend([
            eth_payload.dst_mac,
            eth_payload.src_mac,
            eth_payload.has_header_802_1q,
            eth_payload.header_802_1q,
            eth_payload.ethertype,
            ip_payload.version,
            ip_payload.ihl,
            ip_payload.dscp,
            ip_payload.ecn,
            ip_payload.total_length,
            ip_payload.id_,
            ip_payload.reserved,
            ip_payload.df,
            ip_payload.mf,
            ip_payload.fragment_offset,
            ip_payload.ttl,
            ip_payload.proto,
            ip_payload.header_checksum,
            ip_payload.src,
            ip_payload.dst
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
            transport_payload.reserved,
            transport_payload.cwr,
            transport_payload.ece,
            transport_payload.urg,
            transport_payload.ack,
            transport_payload.psh,
            transport_payload.rst,
            transport_payload.syn,
            transport_payload.fin,
            transport_payload.window,
            transport_payload.checksum,
            transport_payload.urgent_pointer,
            tcp_options.has_max_seg,
            tcp_options.max_seg,
            tcp_options.has_max_win,
            tcp_options.win_scale,
            tcp_options.has_sack_perm,
            tcp_options.has_sack,
            tcp_options.nsack,
            ])
        for s in tcp_options.sack:
            data.extend([
                       s[0],
                       s[1]
                       ])
        for i in range(4 - len(tcp_options.sack)):
            data.extend([0, 0])
        data.extend([
            tcp_options.has_ts,
            tcp_options.ts,
            tcp_options.prev_ts,
            tcp_options.has_user_to,
            tcp_options.user_timeout_gran,
            tcp_options.user_timeout,
            transport_payload.payload_size
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
        X = self.time_series[index, :]
        y = self.time_series[index + 1, :]
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


def run_decode(argv):
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
        if isinstance(i, PCAP):
            point = to_dataset(i)
            if point is not None:
                dataset.append(point)

    dataset = np.array(dataset)
    dataset = split_into_sessions(dataset)
    head = [
            'timestamp',
            'len_original',
            'dst_mac',
            'src_mac',
            'has_header_802_1q',
            'header_802_1q',
            'ethertype',
            'ip_version',
            'ip_ihl',
            'ip_dscp',
            'ip_ecn',
            'ip_total_length',
            'ip_id',
            'ip_reserved',
            'ip_df',
            'ip_mf',
            'ip_fragment_offset',
            'ip_ttl',
            'ip_proto',
            'ip_header_checksum',
            'ip_src',
            'ip_dst',
            'tcp_src',
            'tcp_dst',
            'tcp_seq_no',
            'tcp_ack_no',
            'tcp_data_off',
            'tcp_reserved',
            'tcp_cwr',
            'tcp_ece',
            'tcp_urg',
            'tcp_ack',
            'tcp_psh',
            'tcp_rst',
            'tcp_syn',
            'tcp_fin',
            'tcp_window',
            'tcp_checksum',
            'tcp_urgent_pointer',
            'tcp_opt_has_max_seg',
            'tcp_opt_max_seg',
            'tcp_opt_has_max_win',
            'tcp_opt_win_scale',
            'tcp_opt_has_sack_perm',
            'tcp_opt_has_sack',
            'tcp_opt_nsack',
            'tcp_opt_sack_11',
            'tcp_opt_sack_12',
            'tcp_opt_sack_21',
            'tcp_opt_sack_22',
            'tcp_opt_sack_31',
            'tcp_opt_sack_32',
            'tcp_opt_sack_41',
            'tcp_opt_sack_42',
            'tcp_opt_has_ts',
            'tcp_opt_ts',
            'tcp_opt_prev_ts',
            'tcp_opt_has_user_to',
            'tcp_opt_user_timeout_gran',
            'tcp_opt_user_timeout',
            'payload_size'
            ]
    return pd.DataFrame(dataset, columns=head).sort_values(by=['timestamp', 'ip_id'])


if __name__ == "__main__":
    run_decode(sys.argv).to_csv('inputs.csv')


