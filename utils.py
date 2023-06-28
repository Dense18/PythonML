import numpy as np
from numpy.typing import ArrayLike
from collections import Counter

def most_common_label(y: ArrayLike):
    """
    Returns the most common label from feature [y]
    """
    return Counter(y).most_common(1)[0][0] # Verstaile. Faster than scipy only if y is not a numpy array
    # return st.mode(y, keepdims = False).mode # Much Faster than Counter if y is a numpy array, otherwise slower
    # return np.bincount(y).argmax() # Fastest, best with numpy array. only works if none of the value of y is very large. 