from abc import ABC
from numpy.typing import ArrayLike
import numpy as np

class SupervisedModel(ABC):
    def predict(self, X: ArrayLike):
        """
        Predicts class/regression value for [X]
        """
        raise NotImplementedError
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fits the model based on the training set ([X], [y])
        """
        raise NotImplementedError

    def validate_fit_args(self, X: ArrayLike, y: ArrayLike):
        """
        Validates training set ([X], [y])
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be an nd array. Got a type of {type(X)} instead.")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be an nd array. Got a type of {type(y)} instead.")
        
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-dimensional array. Got {X.ndim}-dimensional array instead.")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1-dimensional array. Got {y.ndim}-dimensional array instead.")
    
class UnSupervisedModel(ABC):
    def predict(self, X: ArrayLike):
        """
        Predicts class/regression value for [X]
        """
        raise NotImplementedError
    
    def fit(self, X: ArrayLike):
        """
        Fits the model based on the training set [X]
        """
        raise NotImplementedError
    
    def validate_fit_args(self, X: ArrayLike):
        """
        Validates training set [X]
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-dimensional array. Got {X.ndim}-dimensional array instead.")