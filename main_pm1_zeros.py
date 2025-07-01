import pandas as pd
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import sys
import matplotlib.pyplot as plt
from utils import num_zeros, get_beautiful_metric, prettyprint_stats, gen_fig, find_max_lr, genetic, find_max_lsvr, find_max_pm1

def run_plots(argv):
    #df = run_decode(argv)
    if len(argv) < 2:
        print(f'Usage: {argv[0]} GROUPED_CSV')
        exit(0)
    df_grouped = pd.read_csv(argv[1], index_col=0)
    print('greedy ±1 choice of column minimizing amt of zeros')
    stats_pm1_zeros, df_pm1_zeros, nreduced_pm1_zeros = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_pm1, num_zeros)
    print('greedy ±1 combination:')
    prettyprint_stats(stats_pm1_zeros)
    zeros_pm1_zeros = gen_fig(df_pm1_zeros, nreduced_pm1_zeros, 'figures/pm1_zeros.png', 'Residual distribution in greedy zeros amount minimization using ±1 coeficients')


if __name__ == '__main__':
    run_plots(sys.argv)
