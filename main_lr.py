import pandas as pd
import sys
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import matplotlib.pyplot as plt
from utils import num_zeros, get_beautiful_metric, prettyprint_stats, gen_fig, find_max_lr

def run_plots(argv):
    #df = run_decode(argv)
    if len(argv) < 2:
        print(f'Usage: {argv[0]} GROUPED_CSV')
        exit(0)
    df_grouped = pd.read_csv(argv[1], index_col=0)
    print('greedy linear regression-based choice of column using D2')
    stats_lr, df_lr, nreduced_lr = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_lr, d2_absolute_error_score)
    print('greedy linear regression-based combination:')
    prettyprint_stats(stats_lr)
    zeros_lr = gen_fig(df_lr, nreduced_lr, 'figures/lr_d2.png', 'Residual distribution in greedy $D^2$ minimization using linear regression')


if __name__ == '__main__':
    run_plots(sys.argv)
