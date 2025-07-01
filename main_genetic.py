import pandas as pd
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import sys
import matplotlib.pyplot as plt
from utils import genetic

def run_plots(argv):
    #df = run_decode(argv)
    if len(argv) < 2:
        print(f'Usage: {argv[0]} GROUPED_CSV')
        exit(0)
    df_grouped = pd.read_csv(argv[1], index_col=0)
    print('run genetic algorithm')
    col = 'tcp_seq_no'
    col_ignore = {col, 'session', 'reverse_session', 'session_y', 'reverse_session_y', 'session_old', 'reverse_session_old'}
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



if __name__ == '__main__':
    run_plots(sys.argv)
