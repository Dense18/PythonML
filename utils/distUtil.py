import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

def euclidean(x1: ArrayLike, x2: ArrayLike, axis: Optional[int | tuple[int]] = None) -> float:
    """
    Calculates the euclidean distance between [X1] and [X2] along the [axis]
    """
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=axis))

def manhattan(x1: ArrayLike, x2: ArrayLike, axis: Optional[int | tuple[int]] = None):
    """
    Calculates the euclidean distance between [X1] and [X2] along the [axis]
    """
    return np.absolute(np.sum((x1 - x2), axis=axis))