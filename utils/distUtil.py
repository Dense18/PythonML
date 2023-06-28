import numpy as np
from numpy.typing import ArrayLike

def euclidean(x1: ArrayLike, x2: ArrayLike) -> float:
    """
    Calculates the euclidean distance between [X1] and [X2]
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan(x1: ArrayLike, x2: ArrayLike):
    """
    Calculates the euclidean distance between [X1] and [X2]
    """
    return np.sqrt(np.absolute(np.sum((x1 - x2))))