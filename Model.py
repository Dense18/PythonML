from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
import numpy as np
from utils.validation import check_consistent_length
from utils.validation import check_X_y
from utils.validation import check_X
class SupervisedModel(ABC):
    @abstractmethod
    def predict(self, X: ArrayLike):
        """
        Predicts class/regression value for [X]
        """
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fits the model based on the training set ([X], [y])
        """
        # self.validate_fit_args(X, y)
        check_X_y(X, y)
        self.n_features_in = X.shape[1]
    
class UnSupervisedModel(ABC):
    @abstractmethod
    def predict(self, X: ArrayLike):
        """
        Predicts class/regression value for [X]
        """
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: ArrayLike):
        """
        Fits the model based on the training set [X]
        """
        check_X(X)
        self.n_features_in = X.shape[1]