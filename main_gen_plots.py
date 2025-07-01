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
    print('greedy linear regression-based choice of column using D2')
    stats_lr, df_lr, nreduced_lr = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_lr, d2_absolute_error_score)
    print('greedy linear regression-based combination:')
    prettyprint_stats(stats_lr)
    zeros_lr = gen_fig(df_lr, nreduced_lr, 'figures/lr_d2.png', 'Residual distribution in greedy $D^2$ minimization using linear regression')
    col = 'tcp_seq_no'
    col_ignore = {col, 'session', 'reverse_session', 'session_y', 'reverse_session_y', 'session_old', 'reverse_session_old'}
    print('run genetic algorithm')
    nzeros_genetic, pop_genetic = genetic(df_grouped)
    #pop_genetic = np.zeros(184, dtype=int)
    #pop_genetic[[104, 136, 30, 86, 44, 89, 43, 59]] = 1
    #nzeros_genetic = 23821
    df_genetic = df_grouped.drop(col_ignore, axis=1)
    col_idx_genetic = np.argwhere(pop_genetic != 0)
    print('genetic resulting combination:')
    for i in col_idx_genetic:
        print(f'col: {df_genetic.columns[i].values[0]:27}, coef: {pop_genetic[i][0]}')
    ytrue_genetic = df_grouped[col]
    comb_genetic = df_genetic @ pop_genetic
    gt_1e3_genetic = (np.abs(comb_genetic - ytrue_genetic) > 1e3).sum()
    btwn_half_1e3_genetic = ((np.abs(comb_genetic - ytrue_genetic) <= 1e3) & (np.abs(comb_genetic - ytrue_genetic) >= 0.5)).sum()
    lt_half_genetic = (np.abs(comb_genetic - ytrue_genetic) < 0.5).sum()
    
    print('genetic nzeros:', lt_half_genetic)
    plt.figure(figsize=(8, 6))
    plt.title('Residual distribution in genetic algorithm')
    bb = plt.bar(['Residual values > 1e3', '0.5 < Residual values < 1e3', 'Residual values < 0.5'], 
                 [gt_1e3_genetic, btwn_half_1e3_genetic, lt_half_genetic])
    plt.bar_label(bb, label_type='edge')
    plt.savefig('figures/genetic.png')

    print('greedy SVR-based choice of column')
    stats_svr, df_svr, nreduced_svr = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_lsvr, d2_absolute_error_score)
    print('greedy SVR-based combination:')
    prettyprint_stats(stats_svr)
    zeros_svr = gen_fig(df_svr, nreduced_svr, 'figures/svr_d2.png', 'Residual distribution in greedy $D^2$ minimization using SVR')
    print('greedy ±1 choice of column')
    stats_pm1_d2, df_pm1_d2, nreduced_pm1_d2 = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_pm1, d2_absolute_error_score)
    print('greedy ±1 combination:')
    prettyprint_stats(stats_pm1_d2)
    zeros_pm1_d2 = gen_fig(df_pm1_d2, nreduced_pm1_d2, 'figures/pm1_d2.png', 'Residual distribution in greedy $D^2$ minimization using ±1 coeficients')
    print('greedy ±1 choice of column minimizing amt of zeros')
    stats_pm1_zeros, df_pm1_zeros, nreduced_pm1_zeros = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_pm1, num_zeros)
    print('greedy ±1 combination:')
    prettyprint_stats(stats_pm1_zeros)
    zeros_pm1_zeros = gen_fig(df_pm1_zeros, nreduced_pm1_zeros, 'figures/pm1_zeros.png', 'Residual distribution in greedy zeros amount minimization using ±1 coeficients')


    plt.figure(figsize=(8, 6))
    plt.title('Total |residual| < 0.5 amount by model')
    bb = plt.bar(['genetic', 'lr+d2', 'svr+d2', 'pm1+d2', 'pm1_zeros'],
                 [lt_half_genetic, zeros_lr, zeros_svr, zeros_pm1_d2, zeros_pm1_zeros])
    plt.bar_label(bb, label_type='edge')
    plt.savefig('figures/total.png')






if __name__ == '__main__':
    run_plots(sys.argv)
