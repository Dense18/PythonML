import numpy as np
from numpy.typing import ArrayLike
from collections import Counter
from typing import Optional

def most_common_label(y: ArrayLike):
    """
    Returns the most common label from feature [y]
    """
    return Counter(y).most_common(1)[0][0] # Verstaile. Faster than scipy only if y is not a numpy array
    # return st.mode(y, keepdims = False).mode # Much Faster than Counter if y is a numpy array, otherwise slower
    # return np.bincount(y).argmax() # Fastest, best with numpy array. only works if y is an array of integers with values that aren't large 

def obtain_bootstrap_idxs(X: ArrayLike, rng: Optional[np.random.Generator] = None, return_oob = True):
    """
    Obtain bootstrap indexes samples from dataset ([X], y)
    
    Returns:
        idxs: indexes used for bootstrap samples
        oob_idxs: indexes used for OOB (Out-Of-Bag). Only provided if return_oob is True
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n_samples = X.shape[0]
    idxs = rng.choice(n_samples, n_samples, replace = True)
    if not return_oob:
        return idxs
    
    oob_idxs = np.setdiff1d(np.arange(n_samples), idxs)
    return idxs, oob_idxs

def obtain_bootstrap_samples(X: ArrayLike, y: ArrayLike, rng: Optional[np.random.Generator] = None, return_oob = True):
    """
    Obtain bootstrap samples from dataset ([X], y)
    
    Returns:
        X_bootstrap: bootstrap samples from X
        X_oob: OOB (Out-Of-Bag) samples from X. Only provided if return_oob is True
        y_bootstrap: bootstrap samples from Y
        y_oob: OOB (Out-Of-Bag) samples from Y. Only provided if return_oob is True
    """
    if return_oob:
        idxs, oob_idxs = obtain_bootstrap_idxs(X, rng, return_oob)
        return X[idxs], X[oob_idxs], y[idxs], y[oob_idxs]
    else:
        idxs = obtain_bootstrap_idxs(X, rng, return_oob)
        return X[idxs], y[idxs]
