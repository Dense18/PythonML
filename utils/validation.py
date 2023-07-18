import numpy as np
from numpy.typing import NDArray


def check_consistent_length(*arrays):
    """
    Check if all given arrays have consistent first dimension.
    """
    lengths = [len(arr) for arr in arrays]
    n_unique = np.unique(lengths)
    if len(n_unique) > 1:
        raise ValueError(
            f"Inconsistent number of samples found: {[l for l in lengths]}"
        )

def check_array(arr: NDArray, 
                *, 
                all_finite: bool = True,
                ensure_2d: bool = True,
                allow_nd: bool = False, 
                min_samples: int = 1,
                min_features: int = 1,
                ):
    """
    Validate input array [arr]
    """
    if arr.ndim == 0:
        raise ValueError(f"Expected an array. Got a scalar {arr} instead")
    
    if all_finite:
        assert_all_finite(arr)
        
    if ensure_2d:
        if arr.ndim != 2:
            raise ValueError(f"Expected 2d array. Got a dimension of {arr.ndim} instead")
    
    if not allow_nd and arr.ndim > 2:
        raise ValueError(f"Expected array with less than 2 dimensions. Got a dimension of {arr.ndim} instead")
    
    if min_samples > 0:
        n_samples = arr.shape[0] # len(arr)
        if n_samples < min_samples:
            raise ValueError(f"Minimum of {min_samples} samples required. Recevied {n_samples} samples instead")
    
    if min_features > 0 and arr.ndim == 2:
        n_features = arr.shape[1]
        if n_features < min_features:
            raise ValueError(f"Minimum of {min_features} features required. Received {n_features} features instead")

def check_X_y(X, 
              y,
              *,
              all_finite: bool = True,
             ):
    """
    Enforce [X] to be 2D and [y] to be 1D with valid structure
    """
    
    check_X(X, all_finite = all_finite)
    check_y(y)
    check_consistent_length(X, y)

def check_X(X: NDArray, 
            *, 
            all_finite: bool = True,
            ):
    """
    Validate [X] array input.
    """
    check_array(
        X,
        all_finite = all_finite,
        ensure_2d = True,
        allow_nd = False, 
        min_samples = 1,
        min_features = 1,
    )
    
def check_y(y: NDArray):
    """
    Validate [y] array input.
    """
    check_array(
        y,
        all_finite = True,
        ensure_2d = False,
        allow_nd = False,
        min_samples = 1,
        min_features = 1
    )
    

def assert_all_finite(arr: NDArray, allow_inf: bool = False, allow_nan: bool = False):
    """
    Assert that all values of the array are finite. Throws ValueError otherwise.
    """
    has_inf_err = False if allow_inf else np.isinf(arr).any()
    if has_inf_err:
        raise ValueError("Expected finite values. Found infinite or too large value(s) on array")
    
    has_nan_err = False if allow_nan else np.isnan(arr).any()
    if has_nan_err:
        raise ValueError("Expected finite values. Found nan value(s) on array")