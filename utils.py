from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import matplotlib.pyplot as plt


def big2i(bs):
    s = 0
    for b in bs:
        s = s*256 + b
    return s


def lit2i(bs):
    s = 0
    for b in bs[::-1]:
        s = s*256 + b
    return s


def i2big(n, n_bytes):
    return (n % 2**(n_bytes * 8)).to_bytes(n_bytes, byteorder='big')


def i2lit(n, n_bytes):
    return (n % 2**(n_bytes * 8)).to_bytes(n_bytes, byteorder='little')


def ip2str(b):
    return '.'.join(map(str, b))


def b2str(b):
    return ' '.join(map(lambda x: f'{x:02x}', b))


def shuffle(b):
    n = len(b) // 2
    b1 = b.copy()
    for i in range(n):
        b1[2*i] = b[i]
        b1[2*i+1] = b[n+i]
    return b1


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
        ngen = 20000
        max_comb = None
        max_zeros = -1
        col_ignore = {col, 'session', 'reverse_session', 'session_y', 'reverse_session_y', 'session_old', 'reverse_session_old'}
        pop = np.zeros(180, dtype=int)
        pop[[104, 136, 30, 86, 44, 89, 43, 59]] = 1
        pop = np.array([pop] * npop).transpose()
        ytrue = np.tile(df[col].to_numpy(), (npop, 1)).transpose()
        df_actual = df.copy().drop(col_ignore, axis=1)
        max_score = float('-inf')
        for gen in range(ngen):
            comb = df_actual @ pop
            scores_zeros = np.sum(np.abs(ytrue-comb) < 0.5, axis=0)
            scores = d2_absolute_error_score(ytrue, comb, multioutput='raw_values') + scores_zeros
            if max(scores) > max_score:
                max_zeros = max(scores_zeros)
                max_score = max(scores)
                max_comb = pop[:, np.argmax(scores_zeros)]
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


