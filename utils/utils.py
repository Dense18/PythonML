from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


def most_common_label(y: ArrayLike):
    """
    Return the most common label from feature [y]
    """
    return Counter(y).most_common(1)[0][0] # Verstaile. Faster than scipy only if y is not a numpy array
    # return st.mode(y, keepdims = False).mode # Much Faster than Counter if y is a numpy array, otherwise slower
    # return np.bincount(y).argmax() # Fastest, best with numpy array. only works if y is an array of integers with values that aren't large 

def bootstrap_idxs(X: ArrayLike, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset [X]
    
    Returns:
        idxs: indexes used for bootstrap samples
    """
    rng = np.random.default_rng(random_state)
    
    n_samples = X.shape[0]
    idxs = rng.choice(n_samples, n_samples, replace = True)
    return idxs

def bootstrap_idxs_with_oob(X: ArrayLike, random_state: Optional[int | np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset [X] with OOB indexes
    
    Returns:
        idxs: indexes used for bootstrap samples
        oob_idxs: indexes used for OOB (Out-Of-Bag). 
    """
    idxs = bootstrap_idxs(X, random_state)
    oob_idxs = np.setdiff1d(np.arange(X.shape[0]), idxs)
    return idxs, oob_idxs


def bootstrap_samples(X: ArrayLike, y: ArrayLike, random_state: Optional[int | np.random.Generator] = None):
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
    
def bootstrap_samples_with_oob(X: ArrayLike, y: ArrayLike, random_state: Optional[int | np.random.Generator] = None):
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