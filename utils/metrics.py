import numpy as np
from numpy.typing import ArrayLike
from utils.validation import check_consistent_length
from typing import Optional

def euclidean(x1: ArrayLike, x2: ArrayLike, axis: Optional[int | tuple[int]] = None) -> float:
    """
    Calculate the euclidean distance between [X1] and [X2] along the [axis]
    where axis is perform during the summation 
    """
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=axis))

def manhattan(x1: ArrayLike, x2: ArrayLike, axis: Optional[int | tuple[int]] = None):
    """
    Calculate the euclidean distance between [X1] and [X2] along the [axis]
    where axis is perform during the summation 
    """
    return np.absolute(np.sum((x1 - x2), axis=axis))

def accuracy_score(y_test: ArrayLike, y_pred: ArrayLike, normalize: bool = True):
    """
    Return the accuracy classification score
    
    if normalize is True: returns the fraction of correctly classfied samples. Otherwise, returns the number of correctly classfied samples.
    """
    check_consistent_length(y_test, y_pred)
    score = y_test == y_pred
    return np.average(score) if normalize else score