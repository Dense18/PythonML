from Model import SupervisedModel
import numpy as np
import math
from collections import Counter
from  numpy.typing import ArrayLike
from typing import Optional
from scipy import stats as st
from utils import most_common_label
class Node:
    """
    A single Node instance of the Decision Tree
    """
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, *, value = None) -> None:
        #Decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = None
        
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
    """
    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = 2,
                 ):
        self.X = None
        self.y = None
        
        self.root = None
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, X: ArrayLike, y: ArrayLike, cur_depth = 0) -> Node:
        """
        Builds the decision tree
        """
        num_samples, num_features = np.shape(X)
        n_labels = len(np.unique(y))
        
        ## Decision Node
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth and n_labels != 1:
            feature_index, threshold, gain, left_indexs, right_indexs = self.best_split(X, y, num_features)
            if gain < 0:
                raise ValueError("Info gain cannot be negative!")
            left = self.build_tree(X[left_indexs,:],y[left_indexs], cur_depth + 1)
            right = self.build_tree(X[right_indexs,:], y[right_indexs], cur_depth + 1)
            return Node(feature_index, threshold, left, right, gain)
            
        ## Leaf Node
        leaf_value = most_common_label(y) #self.most_common_label(y)
        return Node(value = leaf_value)
    
    # def most_common_label(self, y: ArrayLike):
    #     """
    #     Returns the most common label from feature [y]
    #     """
    #     return Counter(y).most_common(1)[0][0]
    #     # return st.mode(y, keepdims = False).mode
    #     # return np.bincount(y).argmax()
        

    def split(self, feature_data: ArrayLike, threshold):
        """
        Returns the left and right indexes of [feature_data] after a split based on the [threshold] 
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
        Returns the best split information based on the dataset ([X], [Y])
        
        Returns in the following order:
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
        for feature_index in range(num_features):
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

    def info_gain(self, parent_y: ArrayLike, left_y: ArrayLike, right_y: ArrayLike) -> float: 
        """
        Computes the information gain value
        """
        parent_entropy = self.entropy(parent_y)
        n_parent = len(parent_y)
        w_left, w_right = len(left_y)/n_parent, len(right_y)/n_parent
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        
        return parent_entropy - (w_left * left_entropy + w_right * right_entropy)
     
    def info_gain_with_parent_entropy(self, parent_entropy: float, n_parent: int, 
                                      left_y: ArrayLike, right_y: ArrayLike) -> float:
        """
        Computes the information gain value given [parent_entropy] value and parent size [n_parent]
        Note: Make sure that [left_y] and [right_y] are mutually exclusive brances from the parent
        """
        w_left, w_right = len(left_y)/n_parent, len(right_y)/n_parent
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        return parent_entropy -(w_left * left_entropy + w_right * right_entropy)
        
    
    def entropy(self, y: ArrayLike): 
        """
        Compute entropy value from [y]
        """
        # Only support numerical values but faster than the method below
        hist = np.bincount(y.astype(int)) #should be integers, does not work with floats
        prob_features = hist / len(y)
        return  -np.sum([p * np.log2(p) for p in prob_features if p > 0])

        # Alternative method
        # Suports numerical and categorical values, but slower than the method above
        
        # _ , counts = np.unique(y, return_counts=True)
        #prob_features = counts / len(y)  
        # return np.sum([-prob * np.log2(prob) for prob in prob_features])
        
    
    def get_thresholds(self, feature_data: ArrayLike): 
        """
        Returns the unique threshold value of [feature_data]
        """
        ## if it is categorical
        # if np.dtype.type in (np.string_, np.object_): # 
        #     return np.unique(feature_data)
        
        ## if it is numerical
        sorted_col = np.sort(feature_data)
        avg_arr = (sorted_col[1:] + sorted_col[:-1]) / 2
        return avg_arr
        
    def fit(self, X: ArrayLike, y: ArrayLike): 
        """
        Builds a decision tree classifier from the training set ([X],[y]) 
        """
        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            X, y = np.array(X), np.array(y)
            
        self.root = self.build_tree(X, y)
        
    def predict(self, X: ArrayLike):
        """
        Predicts class value for X
        """
        if self.root == None:
            raise SystemError("Classifier has not been fitted yet!")
        return np.array([self.traverse(x, self.root) for x in X])
    
    def traverse(self, instance: ArrayLike, node: Node): 
        """
        Traverse down the tree based on the [node] and returns the class value 
        """
        if node.value != None:
            return node.value
        return self.traverse(instance, node.left) if instance[node.feature_index] <= node.threshold \
            else self.traverse(instance, node.right)
    
    
