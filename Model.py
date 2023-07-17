""" Module for abstract Model classes"""

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

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
    def predict(self, X: ArrayLike):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
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
    def predict(self, X: ArrayLike):
        """
        Predict class/regression value for [X]
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: ArrayLike):
        """
        Fit the model from the training dataset [X]
        """
        check_X(X)
        self.n_features_in = X.shape[1]
        