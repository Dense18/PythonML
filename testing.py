import time
import numpy as np
from scipy import stats as st
from collections import Counter

def split(X_column, split_thresh):
    left_idxs = np.array([i for i, row in enumerate(X_column) if row <= split_thresh])
    right_idxs = np.array([i for i, row in enumerate(X_column) if row > split_thresh])
    return left_idxs, right_idxs

def split2(X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs


x_column = np.random.randint(1, 10000, (1000))
# x_column = [np.random.randint(1,1000) for _ in range(1000)]
a_f = lambda y: y + 3
func = lambda x: a_f(x)


# vec_func = np.vectorize(func)
# start = time.time()
# for i in range(1000000):
#     out = np.array([func(x) for j,x in enumerate(x_column)])
# time_takem = time.time() - start
# print(f"first using Counter: {time_takem}")
# print(out)


start = time.time()
aya = np.zeros(1000)
for i in range(1000000):
    vec_func = np.vectorize(func)
    aya = vec_func(x_column)
    # for j, x in enumerate(x_column):
    #     aya[j] = func(x)
time_takem = time.time() - start
print(f"first using  scipy: {time_takem}")
print(aya)