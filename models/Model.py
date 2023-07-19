""" Module for abstract Model classes"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from utils.validation import check_X, check_X_y


class SupervisedModel(ABC):
    """
    Abstract class for Supervised Models
    """
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.n_features_in = 0

    @abstractmethod
    def predict(self, X: NDArray):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray):
        """
        Fit the model from the training dataset ([X], [y])
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
    def __init__(self) -> None:
        self.X = None
        self.n_features_in = 0

    @abstractmethod
    def predict(self, X: NDArray):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: NDArray):
        """
        Fit the model from the training dataset [X]
        """
        check_X(X)
        self.n_features_in = X.shape[1]
        