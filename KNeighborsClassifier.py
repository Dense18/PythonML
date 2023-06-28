from Model import UnSupervisedModel
from collections import Counter
import numpy as np
from numpy.typing import ArrayLike
from utils import most_common_label

class KNeighborsClassifier(UnSupervisedModel):
    """Classifier implementing  k-nearest neighbors vote"""
    
    def __init__(self, n_neighbors:int = 5, algorithm = "euclidean"):
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        
        self.algo_dict = {"euclidean": self.euclidean, "manhattan": self.manhattan}
        self.dist_func = self.algo_dict[algorithm]
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            X, y = np.array(X), np.array(y)
        self.X = X
        self.y = y
    
    def euclidean(self, x1: ArrayLike, x2: ArrayLike) -> float:
        """
        Calculates the euclidean distance between [X1] and [X2]
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan(self, x1: ArrayLike, x2: ArrayLike):
        """
        Calculates the euclidean distance between [X1] and [X2]
        """
        return np.sqrt(np.absolute(np.sum((x1 - x2))))
    
    def _predict(self, x: ArrayLike):
        dist_list = [self.dist_func(x, instance) for instance in self.X]
        sorted_indexs = np.argsort(dist_list)[:self.n_neighbors]
        y_values = self.y[sorted_indexs]
        return most_common_label(y_values)

    def predict(self, X: ArrayLike):
        return np.array([self._predict(x) for x in X])