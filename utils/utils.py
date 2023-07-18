from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import utils.validation as valid


def most_common_label(y: NDArray):
    """
    Return the most common label from feature [y]
    """
    return Counter(y).most_common(1)[0][0] # Verstaile. Faster than scipy only if y is not a numpy array
    # return st.mode(y, keepdims = False).mode # Much Faster than Counter if y is a numpy array, otherwise slower
    # return np.bincount(y).argmax() # Fastest, best with numpy array. only works if y is an array of integers with values that aren't large 

def bootstrap_idxs(X: NDArray, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset [X]
    
    Returns:
        idxs: indexes used for bootstrap samples
    """
    rng = np.random.default_rng(random_state)
    
    n_samples = X.shape[0]
    idxs = rng.choice(n_samples, n_samples, replace = True)
    return idxs

def bootstrap_idxs_with_oob(X: NDArray, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset [X] with OOB indexes
    
    Returns:
        idxs: indexes used for bootstrap samples
        oob_idxs: indexes used for OOB (Out-Of-Bag). 
    """
    idxs = bootstrap_idxs(X, random_state)
    oob_idxs = np.setdiff1d(np.arange(X.shape[0]), idxs)
    return idxs, oob_idxs


def bootstrap_samples(X: NDArray, y: NDArray, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap samples from dataset ([X], y)
    
    Returns:
        X_bootstrap: bootstrap samples from X
        X_oob: OOB (Out-Of-Bag) samples from X. Only provided if return_oob is True
        y_bootstrap: bootstrap samples from Y
        y_oob: OOB (Out-Of-Bag) samples from Y. Only provided if return_oob is True
    """
    idxs = bootstrap_idxs(X, random_state)
    return X[idxs], y[idxs]
    
def bootstrap_samples_with_oob(X: NDArray, y: NDArray, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap samples from dataset ([X], y) with oob samples
    
    Returns:
        X_bootstrap: bootstrap samples from X
        X_oob: OOB (Out-Of-Bag) samples from X. 
        y_bootstrap: bootstrap samples from Y
        y_oob: OOB (Out-Of-Bag) samples from Y.
    """
    idxs, oob_idxs = bootstrap_idxs_with_oob(X, random_state)
    return X[idxs], X[oob_idxs], y[idxs], y[oob_idxs]

def batch_idxs(X: NDArray, batch_size: int, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain [batch_size] samples indexes from [X]
    
    If batch_size is higher than n_rows of X, then n_rows of X will be used instead
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size should be greater than 0. Got a value of {batch_size} instead")
    
    rng = np.random.default_rng(random_state)
    
    n_samples = X.shape[0]
    return rng.choice(n_samples, min(n_samples, batch_size), replace = False)

def batch_samples(X: NDArray, y: NDArray, batch_size: int, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain [batch_size] samples from ([X], [Y])
    
    If batch_size is higher than n_rows of X, then n_rows of X will be used instead
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size should be greater than 0. Got a value of {batch_size} instead")
    valid.check_consistent_length(X, y)
    
    idxs = batch_idxs(X, batch_size, random_state = random_state)
    return X[idxs], y[idxs]