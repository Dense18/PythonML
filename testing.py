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

## Wrapper function
def timer(func):
    def inner_func(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        time_taken = time.time() - start
        print(f"Time taken: {time_taken}")
    return inner_func

## Wrapper function
def timer_repeat(func):
    pass

x_column = np.random.randint(1, 10000, (1000))

a_f = lambda y: y + 3
func = lambda x: a_f(x)


vec_func = np.vectorize(func)
start = time.time()
l = []
for i in range(10000):
    uni = set(x_column)
    aa = np.array([i for i in range(10000) if i not in uni])
time_takem = time.time() - start
print(f"first using Counter: {time_takem}")



start = time.time()

for i in range(10000):
    x2_column = np.arange(10000)
    bb = np.setdiff1d(x2_column, x_column)
time_takem = time.time() - start
print(f"first using  scipy: {time_takem}")

print()
print(len(aa) == len(bb) and all(aa == bb))