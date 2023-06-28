import numpy as np
from numpy.typing import ArrayLike
from collections import Counter

def most_common_label(y: ArrayLike):
        return Counter(y).most_common(1)[0][0]
        # return st.mode(y, keepdims = False).mode
        # return np.bincount(y).argmax()