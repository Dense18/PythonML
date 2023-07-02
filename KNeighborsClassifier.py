from Model import SupervisedModel
from collections import Counter
import numpy as np
from numpy.typing import ArrayLike
import utils.distUtil as distUtil
from utils.utils import most_common_label

class KNeighborsClassifier(SupervisedModel):
    """Classifier implementing  k-nearest neighbors vote"""
    
    def __init__(self, n_neighbors:int = 5, algorithm = "euclidean"):
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        
        self.algo_dict = {"euclidean": distUtil.euclidean, "manhattan": distUtil.manhattan}
        self.dist_func = self.algo_dict[algorithm]
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fits the model based on the training set ([X], [y])
        """
        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            X, y = np.array(X), np.array(y)
        self.X = X
        self.y = y
    
    def _predict(self, x: ArrayLike):
        """
        Predicts class value for instance [x]
        """
        dist_list = [self.dist_func(x, instance) for instance in self.X]
        sorted_indexs = np.argsort(dist_list)[:self.n_neighbors]
        y_values = self.y[sorted_indexs]
        return most_common_label(y_values)

    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        return np.array([self._predict(x) for x in X])