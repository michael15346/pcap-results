import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import sys
from main_decode import run_decode
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import d2_absolute_error_score
pd.options.mode.copy_on_write = True


def num_zeros(y, ypred):
    diff = y - ypred
    diff_norm = diff - diff.median()
    return np.sum(np.abs(diff_norm) < 0.5)


def single_lsvr(df, col, metric, y):
    r = LinearSVR(tol=1e-10, epsilon=1e-4)
    x = np.array(df[col]).reshape(-1, 1)
    r.fit(x, y)
    ypred = r.predict(x)
    norm = metric(y, ypred)
    return (norm, r)


def single_lr(df, col, metric, y):
    r = LinearRegression()
    x = np.array(df[col]).reshape(-1, 1)
    r.fit(x, y)
    ypred = r.predict(x)
    norm = metric(y, ypred)
    return (norm, r)


def find_max_lsvr(df, col, cols_ignore, metric):
    y = df[col]
    maxx = float('-inf')
    e = ThreadPoolExecutor(max_workers=8)
    futures = {e.submit(single_lsvr, df, col, metric, y): col for col in set(df.columns) - cols_ignore}
    for f in as_completed(futures):
        col = futures[f]
        norm, r = f.result()
        if norm > maxx:
            maxx = norm
            max_col = col
            max_model = r
    return (max_col, maxx, max_model.coef_[0], max_model.intercept_[0])


def find_max_lr(df, col, cols_ignore, metric):
    y = df[col]
    maxx = float('-inf')
    e = ThreadPoolExecutor(max_workers=8)
    futures = {e.submit(single_lr, df, col, metric, y): col for col in set(df.columns) - cols_ignore}
    for f in as_completed(futures):
        col = futures[f]
        norm, r = f.result()
        if norm > maxx:
            maxx = norm
            max_col = col
            max_model = r
    return (max_col, maxx, max_model.coef_[0], max_model.intercept_)


def find_max_pm1(df, col, cols_ignore, metric):
    y = df[col]
    maxx = float('-inf')
    for col in set(df.columns) - cols_ignore:
        x = np.array(df[col])
        ypred_p1 = x
        ypred_m1 = -x
        a_list = [1, -1]
        b_list = [-(y - ypred_p1).median(), -(y - ypred_m1).median()]
        metric_list = [
            metric(y, ypred_p1),
            metric(y, ypred_m1)
        ]
        i_norm = np.argmax(metric_list)
        norm = metric_list[i_norm]
        a = a_list[i_norm]
        b = b_list[i_norm]
        if norm > maxx:
            maxx = norm
            max_col = col
            max_a = a
            max_b = b
    return (max_col, maxx, max_a, max_b)


def get_beautiful_metric(df, col, method, metric):
    df = df.copy()
    cols_ignore = {col, 'session', 'reverse_session', 'session_y', 'reverse_session_y', 'session_old', 'reverse_session_old'}
    stats = []
    minn_prev = float('-inf')
    last_reduced = 0
    for nreduced in range(len(df.columns) - 1):
        min_col, minn, a, b = method(df, col, cols_ignore, metric)
        if minn == minn_prev:
            break
        df[f'reduced{nreduced}'] = df[col] - (np.array(df[min_col]) * a + b)
        col = f'reduced{nreduced}'
        cols_ignore.update({f'reduced{nreduced}', min_col})
        stats.append({'col': min_col, 'score': minn, 'a': a, 'b': b})
        minn_prev = minn
        last_reduced = nreduced
    return stats, df, last_reduced


def prettyprint_stats(stats):
    for i in stats:
        print(f'col: {i['col']:27}, metric: {i['score']:10.4e}, a: {i['a']:10.2e}, b: {i['b']:10.2e}')


def breathe_with_me(col1, col2, pmut):
    rng = np.random.default_rng()
    mask1 = rng.integers(0,2, (len(col1))) == 1
    mmask1 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut1 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask2 = rng.integers(0,2, (len(col1))) == 1
    mmask2 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut2 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask3 = rng.integers(0,2, (len(col1))) == 1
    mmask3 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut3 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask4 = rng.integers(0,2, (len(col1))) == 1
    mmask4 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut4 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask5 = rng.integers(0,2, (len(col1))) == 1
    mmask5 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut5 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask6 = rng.integers(0,2, (len(col1))) == 1
    mmask6 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut6 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask7 = rng.integers(0,2, (len(col1))) == 1
    mmask7 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut7 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    mask8 = rng.integers(0,2, (len(col1))) == 1
    mmask8 = rng.choice([0,1],col1.shape, p=[1-pmut, pmut]) == 1
    mut8 = rng.choice([-1, 0, 1], col1.shape, p=[pmut / 3, pmut / 3, 1 - (2 * pmut / 3)])
    child1 = np.where(mmask1, mut1, np.where(mask1, col1, col2))
    child2 = np.where(mmask2, mut2, np.where(mask2, col1, col2))
    child3 = np.where(mmask3, mut3, np.where(mask3, col1, col2))
    child4 = np.where(mmask4, mut4, np.where(mask4, col1, col2))
    child5 = np.where(mmask5, mut5, np.where(mask5, col1, col2))
    child6 = np.where(mmask6, mut6, np.where(mask6, col1, col2))
    child7 = np.where(mmask7, mut7, np.where(mask7, col1, col2))
    child8 = np.where(mmask8, mut8, np.where(mask8, col1, col2))
    return [child1, child2, child3, child4, child5, child6, child7, child8]
    # Четверо детей. Плюс ещё четверо детей. Плюс ещё четверо детей.
    # В общем, делаем детей. Выполняем, так сказать, госзаказ
    # на детей. Меня беспокоит вопрос: куда деть детей?
    #
    # Что касается детей, то я их с удовольствием ем
    # каждый день, в свежем либо охлажденном виде.
    # И сегодня тоже.


def genetic(df, col='tcp_seq_no'):
    try:
        npop = 32
        mutation_prob = 0.0005
        ngen = 500
        max_comb = None
        max_zeros = -1
        col_ignore = {col, 'session', 'reverse_session', 'session_y', 'reverse_session_y', 'session_old', 'reverse_session_old'}
        pop = np.zeros(184, dtype=int)
        pop[[104, 136, 30, 86, 44, 89, 43, 59]] = 1
        ytrue = np.tile(df[col].to_numpy(), (npop, 1)).transpose()
        df_actual = df.copy().drop[col_ignore]
        max_score = float('-inf')
        for gen in range(ngen):
            comb = df_actual @ pop
            scores_zeros = np.sum(np.abs(ytrue-comb) < 0.5, axis=0)
            scores = d2_absolute_error_score(ytrue, comb, multioutput='raw_values') + scores_zeros
            print(scores)
            if max(scores) > max_score:
                max_zeros = max(scores_zeros)
                max_score = max(scores)
                max_comb = pop[:, np.argmax(scores_zeros)]
                print('max zeros: ', max_zeros)
                mutation_prob /= 1.1
                mutation_prob = max(mutation_prob, 1e-6)
            else:
                mutation_prob *= 1.1
                mutation_prob = min(mutation_prob, 1 - 1e-8)
            ind = np.argpartition(scores, -npop//4)[-npop//4:]
            ind = ind[np.argsort(scores[ind])][::-1]
            off = []
            for i in range(0, npop // 4, 2):
                off.extend(breathe_with_me(pop[:, ind[i]], pop[:, ind[i+1]], mutation_prob))
            print(np.array(off).shape)
            print('lr', mutation_prob)
            pop = np.array(off).transpose().astype(int)
            max_score = max(max_score, max(scores))
        return max_zeros, max_comb
    except KeyboardInterrupt:
        return max_zeros, max_comb


def gen_fig(df, nreduced, file, title):
    col = df[f'reduced{nreduced}']
    gt_1e3 = (np.abs(col) > 1e3).sum()
    btwn_half_1e3 = ((np.abs(col) <= 1e3) & (np.abs(col) >= 0.5)).sum()
    lt_half = (np.abs(col) < 0.5).sum()
    plt.figure(figsize=(8, 6))
    plt.title(title)
    bb = plt.bar(['Residual values > 1e3', '0.5 < Residual values < 1e3', 'Residual values < 0.5'], 
                 [gt_1e3, btwn_half_1e3, lt_half])
    plt.bar_label(bb, label_type='edge')
    plt.savefig(file)
    return lt_half


def run_plots(argv):
    df = run_decode(argv)
    df = pd.read_csv('entire_input.csv')
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

    #df_grouped.to_csv('grouped.csv')
    #df_grouped = pd.read_csv('grouped.csv').drop('Unnamed: 0.1', axis=1)
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

    print('greedy linear regression-based choice of column using D2')
    stats_lr, df_lr, nreduced_lr = get_beautiful_metric(df_grouped, 'tcp_seq_no', find_max_lr, d2_absolute_error_score)
    print('greedy linear regression-based combination:')
    prettyprint_stats(stats_lr)
    zeros_lr = gen_fig(df_lr, nreduced_lr, 'figures/lr_d2.png', 'Residual distribution in greedy $D^2$ minimization using linear regression')
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
