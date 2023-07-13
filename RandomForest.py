from numpy.typing import ArrayLike
from DecisionTreeClassifier import DecisionTreeClassifier
from Model import SupervisedModel
from sklearn.utils.validation import NotFittedError
from typing import Optional
import numpy as np
from utils.utils import most_common_label
from utils.utils import obtain_bootstrap_samples
from utils.metrics import accuracy_score

class RandomForest(SupervisedModel):
    """
    Random Forest classifier
    """
    def __init__(self, 
                 n_estimators: int = 10,
                 min_samples_split: int = 2,
                 max_depth: int = np.iinfo(np.int32).max,
                 max_features: Optional[int] = None,
                 bootstrap: bool = True,
                 compute_oob: bool = False,
                 random_state: Optional[int|np.random.Generator] = None
                 ):
        
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.compute_oob = compute_oob
        self.rng = np.random.default_rng(random_state)
        self.max_features = max_features 
        self.bootstrap = bootstrap
        
        self.n_features_in = None
        self.max_features_ = None
        self.trees = None
        
        self.oob_score = None
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Builds a decision tree classifier from the training set ([X],[y]) 
        """
        super().fit(X, y)
        self.n_features_in = X.shape[0]
        self.max_features_ = self.n_features_in if self.max_features is None else self.max_features 
        
        total_oob_err = 0
        self.trees = []
        for _ in range(self.n_estimators):
            tree_cls = DecisionTreeClassifier(
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth,
                max_features = self.max_features_,
                random_state = self.rng
            )
            
            if not self.bootstrap:
                tree_cls.fit(X_boot, y_boot)
                self.trees.append(tree_cls)
                continue
            
            if self.compute_oob: # Dont compute oob if unnecessary for efficiency
                X_boot, X_oob, y_boot, y_oob = obtain_bootstrap_samples(X, y, return_oob = True)
            else:
                X_boot, y_boot= obtain_bootstrap_samples(X, y, return_oob = False)    

            tree_cls.fit(X_boot, y_boot)
            self.trees.append(tree_cls)
            
            if self.compute_oob and self.bootstrap and X_oob.shape[0] > 0:
                y_pred = tree_cls.predict(X_oob)
                total_oob_err += accuracy_score(y_oob, y_pred)
        
        if self.compute_oob and self.bootstrap:
            self.oob_score = total_oob_err / self.n_estimators
            
    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        if self.trees is None:
            raise NotFittedError("Random Forest Classifier has not been fitted yet!")
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T
        return np.array([most_common_label(pred) for pred in tree_preds])