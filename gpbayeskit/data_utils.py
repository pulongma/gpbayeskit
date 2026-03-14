import numpy as np


def data_split(X, y, train_frac=0.8, seed=None):
    rng = np.random.default_rng(seed)

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    n = X.shape[0]
    n_train = int(np.floor(train_frac * n))

    perm = rng.permutation(n)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    return (
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )