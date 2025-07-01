import pandas as pd
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import sys
import matplotlib.pyplot as plt
from utils import num_zeros, get_beautiful_metric, prettyprint_stats, gen_fig, find_max_lsvr

def run_plots(argv):
    #df = run_decode(argv)
    if len(argv) < 2:
        print(f'Usage: {argv[0]} GROUPED_CSV')
        exit(0)
    df_grouped = pd.read_csv(argv[1], index_col=0)
    print('greedy SVR-based choice of column')
    stats_svr, df_svr, nreduced_svr = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_lsvr, d2_absolute_error_score)
    print('greedy SVR-based combination:')
    prettyprint_stats(stats_svr)
    zeros_svr = gen_fig(df_svr, nreduced_svr, 'figures/svr_d2.png', 'Residual distribution in greedy $D^2$ minimization using SVR')


if __name__ == '__main__':
    run_plots(sys.argv)
