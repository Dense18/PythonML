from abc import ABC
from numpy.typing import ArrayLike
class Model(ABC):
    def predict(self, X: ArrayLike):
        """
        Predicts class/regression value for X
        """
        raise NotImplementedError
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fits the model based on the training set ([X], [y])
        """
        raise NotImplementedError