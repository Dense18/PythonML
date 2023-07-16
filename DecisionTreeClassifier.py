import math
from typing import Optional, Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import NotFittedError

from Model import SupervisedModel
from utils.utils import most_common_label


class Node:
    """
    A single Node instance of the Decision Tree
    
    Parameters
    ----------
    feature_index:
        Index of the features used for comparison
        
    threshold:
        Threshold value used to split the Node
    
    left:
        Left Node
    
    right:
        Right Node
    
    info_gain:
        Information gain value of the current split based on the Left and Right Node
    
    value:
        Classification value of the Node
        
        Exclusively used for Leaf Node
    
    Attributes
    ----------
    
    Same as the variables in [Parameters]
    """
    
    def __init__(self, 
                 feature_index: int = None, 
                 threshold: float = None,
                 left: Self = None, 
                 right: Self = None, 
                 info_gain: float = None, 
                 *, 
                 value: Optional[int] = None
                 ):
        #Decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        #Leaf Node
        self.value = value
    
    def is_leaf(self) -> bool:
        """
        Identify if the node is a leaf node
        """
        return self.value != None
    
class DecisionTreeClassifier(SupervisedModel):
    """
        Decision Tree Classifier. Only supports numerical values with numpy array inputs.
        
        Parameters
        ----------
        min_samples_split: 
            Minimum number of samples required to split an internal Node
        
        max_depth: 
            Maximum depth of the tree 
        
        max_features: 
            Number of features to consider when looking for the best split

            Setting max_features to None indicates that all features of the training
            dataset will be used
            
        random_state:
            Value to control the randomness of the model
        
        Attributes
        ----------
        All variables in [Parameters]
        
        max_features_: int
            Inferred value of max features
        
        rng: Generator
            RNG Generator used for randomness 
    """
    
    def __init__(self,
                 *,
                 min_samples_split: int = 2,
                 max_depth: int = np.iinfo(np.int32).max,
                 max_features: Optional[int] = None,
                 random_state: Optional[int | np.random.Generator] = None
                 ):
        
        self.validate_param(
            min_samples_split = min_samples_split,
            max_depth = max_depth,
            max_features = max_features,
            random_state = random_state
        )
        
        self.root = None
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        self.max_features_ = None
    
    def fit(self, X: ArrayLike, y: ArrayLike): 
        """
        Build a decision tree classifier from the training set ([X],[y]) 
        """
        super().fit(X, y)
        self.n_features_in = X.shape[1]
        self.max_features_ = self.n_features_in if self.max_features is None else self.max_features   
        self.root = self.build_tree(X, y)
    
    def traverse(self, instance: ArrayLike, node: Node): 
        """
        Traverse down the tree based on the [node] and returns the class value 
        """
        if node.value != None:
            return node.value
        return self.traverse(instance, node.left) if instance[node.feature_index] <= node.threshold \
            else self.traverse(instance, node.right)
    
    def predict(self, X: ArrayLike):
        """
        Predict class value for [X]
        """
        if self.root is None:
            raise NotFittedError("Classifier has not been fitted yet!")
        return np.array([self.traverse(x, self.root) for x in X])   
    
    
    ###### Tree building ######
    
    
    def build_tree(self, X: ArrayLike, y: ArrayLike, cur_depth = 0) -> Node:
        """
        Build the decision tree
        """
        num_samples = X.shape[0]
        n_labels = len(np.unique(y))
        
        ## Decision Node
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth and n_labels != 1:
            feature_index, threshold, gain, left_indexs, right_indexs = self.best_split(X, y, self.max_features_)
            if gain < 0:
                raise ValueError("Info gain cannot be negative!")
            left = self.build_tree(X[left_indexs,:], y[left_indexs], cur_depth + 1)
            right = self.build_tree(X[right_indexs,:], y[right_indexs], cur_depth + 1)
            return Node(feature_index, threshold, left, right, gain)
            
        ## Leaf Node
        leaf_value = most_common_label(y)
        return Node(value = leaf_value)
        
    def split(self, feature_data: ArrayLike, threshold):
        """
        Return the left and right indexes of [feature_data] after a split based on the [threshold] 
        """
        ## Categorial variable
        # left_idxs = np.argwhere(feature_data == split_thresh).flatten()
        # right_idxs = np.argwhere(feature_data != split_thresh).flatten()
        
        ## Continuous variable
        left_idxs = np.argwhere(feature_data <= threshold).flatten()
        right_idxs = np.argwhere(feature_data > threshold).flatten()
        return left_idxs, right_idxs
    
    def best_split(self, X: ArrayLike, y: ArrayLike, num_features: int):
        """
        Return the best split information based on the dataset ([X], [Y]) with random subset of [num_features] chosen
        
        Return in the following order:
            split_index: feature index of the split
            split_threshhold: split threshold
            max_gain: information gain value based on the the best split
            best_left_idx: list of left indexes for the left array after the split
            best_right_idx: list of right indexes for the left array after the split
        
        """
        split_threshold = None
        split_index = None
        max_gain = -math.inf
        
        best_left_idx = None
        best_right_idx = None
        
        y_entropy, y_size = self.entropy(y), len(y)
        
        features_idx_arr = self.rng.choice(self.n_features_in, min(self.n_features_in, num_features), replace = False) 
        
        for feature_index in features_idx_arr: #for feature_index in range(num_features):
            feature_data = X[:,feature_index] 
            possible_thresholds = np.unique(feature_data) #self.get_thresholds(features)
            for threshold in possible_thresholds:
                left_indexs, right_indexs = self.split(feature_data, threshold)
                if len(left_indexs) == 0 or len(right_indexs) == 0:
                    continue
                
                left_y, right_y = y[left_indexs], y[right_indexs]
                gain = self.info_gain_with_parent_entropy(y_entropy, y_size, left_y, right_y)
                if gain > max_gain:
                    max_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
                    best_left_idx = left_indexs
                    best_right_idx = right_indexs
                    
        return split_index, split_threshold, max_gain, best_left_idx, best_right_idx   

    def get_thresholds(self, feature_data: ArrayLike): 
        """
        Return the unique threshold value of [feature_data]
        """
        ## if it is categorical
        # if np.dtype.type in (np.string_, np.object_): 
        #     return np.unique(feature_data)
        
        ## if it is numerical
        sorted_col = np.sort(feature_data)
        avg_arr = (sorted_col[1:] + sorted_col[:-1]) / 2
        return avg_arr
    
    
    ###### Calculation ######
    
     
    def info_gain(self, parent_y: ArrayLike, left_y: ArrayLike, right_y: ArrayLike) -> float: 
        """
        Compute the information gain value
        """
        parent_entropy = self.entropy(parent_y)
        n_parent = len(parent_y)
        w_left, w_right = len(left_y)/n_parent, len(right_y)/n_parent
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        
        return parent_entropy - (w_left * left_entropy + w_right * right_entropy)
     
    def info_gain_with_parent_entropy(self, parent_entropy: float, n_parent: int, 
                                      left_y: ArrayLike, right_y: ArrayLike) -> float:
        """
        Compute the information gain value given [parent_entropy] value and parent size [n_parent]
        Much more efficient than info_gain(...) counterpart
        
        Note: 
            Ensure that [left_y] and [right_y] are mutually exclusive brances from the parent
        """
        w_left, w_right = len(left_y)/n_parent, len(right_y)/n_parent
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        return parent_entropy -(w_left * left_entropy + w_right * right_entropy)
        
    def entropy(self, y: ArrayLike): 
        """
        Compute entropy value from [y]
        """
        # Only support numerical values with a non-large values but faster than the method below
        hist = np.bincount(y.astype(int)) #should be integers, does not work with floats
        prob_features = hist / len(y)
        return  -np.sum([p * np.log2(p) for p in prob_features if p > 0])

        # Alternative method
        # Suports numerical and categorical values, but slower than the method above
        
        # _ , counts = np.unique(y, return_counts=True)
        #prob_features = counts / len(y)  
        # return np.sum([-prob * np.log2(prob) for prob in prob_features])
    
           
    ###### Validation ######

    
    def validate_param(self, min_samples_split, max_depth, max_features, random_state):
        """
        Validate parameter arguments
        """
        
        if min_samples_split <= 1:
            raise ValueError(f"min_samples_split should be greater than 2. Got {min_samples_split} instead.")
        
        if max_depth < 0:
            raise ValueError(f"min_samples_split should be positive. Got {max_depth} instead.")
        
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features int value should be greater than 1. Got {max_features} instead.")  
    
        if isinstance(random_state, int) and random_state < 0:
            raise ValueError(f"random_state integer value should be greater than 0. Got a value of {random_state} instead.")
        
        
    ###### Text presentation ######
    
    
    def print_tree(self, spacing = 3, max_depth = 10, export = False):
        """
        Print a text representation of the fitted decision tree.
         
        Set "export = True" to return a tree str representation
        """
        
        if self.root == None:
            print("Decision Tree has not been fitted")
            return

        def _print_tree(node: Node, depth = 1):
            indent = ("|" + " " * spacing) * depth
            indent = indent[:-spacing] + "-" * spacing
            
            output = ""
            if depth > max_depth + 1:
                output += f"{indent} Trunctuated branch\n"
                return output
            if node.value != None:
                output += f"{indent} class: {node.value}\n"
                return output
            
            output += f"{indent} feature_{node.feature_index} <= {node.threshold}\n"
            output += _print_tree(node.left, depth = depth + 1)
            
            output += f"{indent} feature_{node.feature_index} >  {node.threshold}\n"
            output +=  _print_tree(node.right, depth = depth + 1)
            
            return output
        
        output = _print_tree(self.root)
        return output if export else print(output)
    
    def __str__(self) -> str:
        return self.print_tree(export = True)

        
        
        
    
