import time
import numpy as np


def split(X_column, split_thresh):
    left_idxs = np.array([i for i, row in enumerate(X_column) if row <= split_thresh])
    right_idxs = np.array([i for i, row in enumerate(X_column) if row > split_thresh])
    return left_idxs, right_idxs

def split2(X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs


x_column = np.random.randint(1, 10000, (1000))
x_column2 = x_column.copy()

start = time.time()
for i in range(1000000):
    np.power(x_column, 2)
time_takem = time.time() - start
print(f"first using list comprehension: {time_takem}")

start = time.time()
for i in range(1000000):
    x_column2
time_takem = time.time() - start
print(f"first using  argwhere: {time_takem}")