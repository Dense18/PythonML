from abc import ABC
from numpy.typing import ArrayLike

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