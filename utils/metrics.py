from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from utils.validation import check_consistent_length


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

def mse(y_test: ArrayLike, y_pred: ArrayLike, squared: bool = True) -> float:
    """
    Calculates Mean Squared Error value
    
    Args:
    --------
    y_test: Ground truth target values
    
    y_pred: Pedicted target values
    
    squared:
        If True, return MSE, else returns RMSE
    """
    check_consistent_length(y_test, y_pred)
    err =  np.average((y_test - y_pred) ** 2)
    
    if not squared:
        err = np.sqrt(err)
    
    return err

def rmse(y_test: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates Root Mean Squared Error value
    
    Convenience function, similar to calling :term:'mse' function
    with 'squared' param set to False
    
    Args:
    --------
    y_test: Ground truth target values
    
    y_pred: Pedicted target values
    """
    return mse(y_test, y_pred, squared = False)
