import sys
from pathlib import Path
from utils import lit2i
from const import PCAP_FILE_HEADER_LEN, PCAP_PACKET_HEADER_LEN
from encoder import Decoder, PCAP, IPv4, TCP
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def run_convert(argv):
    if len(argv) < 2:
        print(f"Usage: {argv[0]} DECODED_CSV")
    df = pd.read_csv(argv[1], index_col=0)
    df['session'] = df['tcp_src'].astype(str) + '_' + df['tcp_dst'].astype(str) + '_' + \
        df['ip_src'].astype(str) + '_' + df['ip_dst'].astype(str)
    df['reverse_session'] = df['tcp_dst'].astype(str) + '_' + df['tcp_src'].astype(str) + '_' + \
        df['ip_dst'].astype(str) + '_' + df['ip_src'].astype(str)
    sids = df.session.unique()
    df_grouped = pd.DataFrame()
    for sid in sids:
        sdf = df[df.session == sid].convert_dtypes()
        nseq = sdf[:-1]
        nseq.index = sdf.index[1:]
        nseq.columns = nseq.columns + '_old'
        nseq = nseq.convert_dtypes()
        sdf2 = df[(df.reverse_session == sid) | (df.session == sid)].convert_dtypes()
        sdf2 = pd.merge_asof(sdf, sdf2, left_by='session',
                             right_by='reverse_session', left_on='timestamp',
                             right_on='timestamp', direction='backward',
                             )
        sdf2.index = sdf.index
        sdf2 = sdf2.filter(regex='_y$')
        sdf = pd.concat([sdf, sdf2, nseq], axis=1, copy=False).dropna()
        if not sdf.empty:
            df_grouped = pd.concat([df_grouped, sdf], copy=False)
            print(df_grouped)
            return df_grouped
    return df_grouped


if __name__ == "__main__":
    run_convert(sys.argv).to_csv('grouped.csv', index=False)
