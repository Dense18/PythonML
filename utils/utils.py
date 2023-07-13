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

def obtain_bootstrap_idxs_r(X: ArrayLike, rng: Optional[np.random.Generator] = None, return_oob = True):
    """
    Convenience function that calls obtain_bootstrap_idxs(...) or obtain_bootstrap_idxs_with_oob(...)
    based on the [return_oob] parameter
    """
    return obtain_bootstrap_idxs_with_oob(X, rng) if return_oob else obtain_bootstrap_idxs(X, rng)
    

def obtain_bootstrap_idxs(X: ArrayLike, rng: Optional[np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset ([X], y)
    
    Returns:
        idxs: indexes used for bootstrap samples
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n_samples = X.shape[0]
    idxs = rng.choice(n_samples, n_samples, replace = True)
    return idxs

def obtain_bootstrap_idxs_with_oob(X: ArrayLike, rng: Optional[np.random.Generator] = None):
    """
    Obtain bootstrap indexes samples from dataset ([X], y) with OOB indexes
    
    Returns:
        idxs: indexes used for bootstrap samples
        oob_idxs: indexes used for OOB (Out-Of-Bag). 
    """
    idxs = obtain_bootstrap_idxs(X, rng)
    oob_idxs = np.setdiff1d(np.arange(X.shape[0]), idxs)
    return idxs, oob_idxs

def obtain_bootstrap_samples_r(X: ArrayLike, 
                               y: ArrayLike, 
                               rng: Optional[np.random.Generator] = None, 
                               return_oob = True):
    """
    Convenience function that calls obtain_bootstrap_samples(...) or obtain_bootstrap_samples_with_oob(...)
    based on the [return_oob] parameter
    """
    return obtain_bootstrap_samples_with_oob(X, y, rng) if return_oob else obtain_bootstrap_samples(X, y, rng)


def obtain_bootstrap_samples(X: ArrayLike, y: ArrayLike, rng: Optional[np.random.Generator] = None):
    """
    Obtain bootstrap samples from dataset ([X], y)
    
    Returns:
        X_bootstrap: bootstrap samples from X
        X_oob: OOB (Out-Of-Bag) samples from X. Only provided if return_oob is True
        y_bootstrap: bootstrap samples from Y
        y_oob: OOB (Out-Of-Bag) samples from Y. Only provided if return_oob is True
    """
    idxs = obtain_bootstrap_idxs(X, rng)
    return X[idxs], y[idxs]
    
def obtain_bootstrap_samples_with_oob(X: ArrayLike, y: ArrayLike, rng: Optional[np.random.Generator] = None):
    """
    Obtain bootstrap samples from dataset ([X], y) with oob samples
    
    Returns:
        X_bootstrap: bootstrap samples from X
        X_oob: OOB (Out-Of-Bag) samples from X. 
        y_bootstrap: bootstrap samples from Y
        y_oob: OOB (Out-Of-Bag) samples from Y.
    """
    idxs, oob_idxs = obtain_bootstrap_idxs_with_oob(X, rng)
    return X[idxs], X[oob_idxs], y[idxs], y[oob_idxs]