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


vec_func = np.vectorize(func)
start = time.time()
l = []
for i in range(1000000):
    l = []
    for i in range(1000):
        l.append(i)
    l = np.array(l)
time_takem = time.time() - start
print(f"first using Counter: {time_takem}")



start = time.time()

for i in range(1000000):
    aya = np.zeros(1000)
    for i in range(1000):
        aya[i] = i
time_takem = time.time() - start
print(f"first using  scipy: {time_takem}")