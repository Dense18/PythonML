from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from utils.validation import check_consistent_length, check_X, check_X_y


class SupervisedModel(ABC):
    """
    Abstract class for Supervised Models
    """
    @abstractmethod
    def predict(self, X: ArrayLike):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fit the model based on the training set ([X], [y])
        """
        # self.validate_fit_args(X, y)
        check_X_y(X, y)
        self.X = X
        self.y = y
        self.n_features_in = X.shape[1]
    
class UnSupervisedModel(ABC):
    """
    Abstract class for Unsupervised Models
    """
    @abstractmethod
    def predict(self, X: ArrayLike):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: ArrayLike):
        """
        Fit the model based on the training set [X]
        """
        check_X(X)
        self.n_features_in = X.shape[1]