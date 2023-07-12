import numpy as np

def check_consistent_length(*arrays):
    """
    Checks if all given arrays have consistent first dimension.
    """
    lengths = [len(arr) for arr in arrays]
    n_unique = np.unique(lengths)
    if len(n_unique) > 1:
        raise ValueError(
            f"Inconsistent number of samples found: {[l for l in lengths]}"
        )