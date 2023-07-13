import numpy as np
from numpy.typing import ArrayLike
from utils.validation import check_consistent_length

def accuracy_score(y_test: ArrayLike, y_pred: ArrayLike, normalize: bool = True):
    """
    Return the accuracy classification score
    
    if normalize is True: returns the fraction of correctly classfied samples. Otherwise, returns the number of correctly classfied samples.
    """
    check_consistent_length(y_test, y_pred)
    score = y_test == y_pred
    return np.average(score) if normalize else score