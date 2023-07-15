import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import NotFittedError

from Model import SupervisedModel
from utils.metrics import euclidean, manhattan
from utils.utils import most_common_label


class KNeighborsClassifier(SupervisedModel):
    """
    Classifier implementing k-nearest neighbors vote
    
    Paramaters
    ----------
    n_neighbors:
        Number of neighbors to compare
    
    dist_metric: {"euclidean", "manhattan"}
        Metrics to calculate distance between points
        
        "euclidean": Peform euclidean method. 
        
        "manhattan": Perform manhttan method

        If provided argument is not on the supported values, "euclidean" will be used
    
    Attributes
    ----------
    
    All variables in [Parameters]
    
    X: Array
        Independent variables on the training dataset
    
    Y: Array
        Dependent variable on the training dataset

    dist_func:
        Distance Function based on [dist_metric] value
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
        self.DIST_DICT = {"euclidean": euclidean, "manhattan": manhattan}
        self.dist_func = self.DIST_DICT.get(dist_metric, "euclidean")
       
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fit the model based on the training set ([X], [y])
        """
        super().fit(X, y)  
        self.X, self.y = X, y
        self.n_features_in = X.shape[1]
    
    def _predict(self, x: ArrayLike):
        """
        Predict class value for instance [x]
        """
        dist_list = np.array([self.dist_func(x, instance) for instance in self.X])
        sorted_indexs = np.argsort(dist_list)[:self.n_neighbors]
        y_values = self.y[sorted_indexs]
        return most_common_label(y_values)

    def predict(self, X: ArrayLike):
        """
        Predict class value for [X]
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