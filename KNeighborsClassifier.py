from Model import SupervisedModel
from collections import Counter
import numpy as np
from numpy.typing import ArrayLike
import utils.distUtil as distUtil
from utils.utils import most_common_label
from sklearn.utils.validation import NotFittedError

class KNeighborsClassifier(SupervisedModel):
    """
    Classifier implementing k-nearest neighbors vote
    
    Paramaters:
    ----------
    n_neighbors:
        Number of neighbors to compare
    
    dist_metric: {"euclidean", "manhattan"}
        Metrics to calculate distance between points
        
        "euclidean": Peform euclidean method. 
        
        "manhattan": Perform manhttan method

        If given argument is not on the supported dist_metric methods, "k++" will be used
    
    
    """
    
    def __init__(self, 
                 n_neighbors: int = 5, 
                 *,
                 dist_metric: str = "euclidean"
                 ):
        
        self.validate(
            n_neighbors = n_neighbors,
        )
        
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        
        self.dist_metric = dist_metric
        self.dist_dict = {"euclidean": distUtil.euclidean, "manhattan": distUtil.manhattan}
        self.dist_func = self.dist_dict.get(dist_metric, "euclidean")
       
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fits the model based on the training set ([X], [y])
        """
        super().fit(X, y)  
        self.X, self.y = X, y
        self.n_features_in = X.shape[1]
    
    def _predict(self, x: ArrayLike):
        """
        Predicts class value for instance [x]
        """
        dist_list = np.array([self.dist_func(x, instance) for instance in self.X])
        sorted_indexs = np.argsort(dist_list)[:self.n_neighbors]
        y_values = self.y[sorted_indexs]
        return most_common_label(y_values)

    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        if self.X is None:
            raise NotFittedError("Classifier has not beed fiited yet!")
        return np.array([self._predict(x) for x in X])
    
    
    ####### Validation ######
    
    
    def validate(self, n_neighbors):
        """
        Validate provided arguments
        """
        if n_neighbors < 1:
           raise ValueError(f"n_neighbors should be greater than 0. Got {n_neighbors} instead.")