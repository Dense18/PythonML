from Model import Model
import numpy as np
import math
from collections import Counter

class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, *, value = None) -> None:
        #Deicision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = None
        
        #Leaf Node
        self.value = value
    
    def is_leaf(self) -> bool:
        return self.value != None
    
class DecisionTreeClassifier(Model):
    def __init__(self,
                 min_samples_split = 2,
                 max_depth = 2,
                 n_features = None
                 ):
        self.X = None
        self.y = None
        
        self.root = None
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, X, y, cur_depth = 0) -> Node:
        num_samples, num_features = np.shape(X)
        n_labels = len(np.unique(y))
        
        ## Decision Node
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth and n_labels != 1:
            feature_index, threshold, gain, left_indexs, right_indexs = self.best_split(X, y, num_samples, num_features)
            if gain < 0:
                raise ValueError("Info gain cannot be negative!")
            left = self.build_tree(X[left_indexs,:],y[left_indexs], cur_depth + 1)
            right = self.build_tree(X[right_indexs,:], y[right_indexs], cur_depth + 1)
            return Node(feature_index, threshold, left, right, gain)
            
        ## Leaf Node
        leaf_value = self.most_common_label(y)
        return Node(value = leaf_value)
    
    def most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def best_split(self, X, y, num_samples, num_features):
        split_threshold = None
        split_index = None
        max_gain = -math.inf
        
        best_left_idx = None
        best_right_idx = None
        
        y_entropy, y_size = self.entropy(y), len(y)
        for feature_index in range(num_features):
            features = X[:,feature_index] 
            possible_thresholds = np.unique(features)
            for threshold in possible_thresholds:
                left_indexs, right_indexs = self.split(features, threshold)
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

    def info_gain(self, parent_y, left_y, right_y) -> float: 
        parent_entropy = self.entropy(parent_y)
        n_parent = len(parent_y)
        w_left, w_right = len(left_y)/n_parent, len(right_y)/n_parent
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        
        return parent_entropy - (w_left * left_entropy + w_right * right_entropy)
     
    def info_gain_with_parent_entropy(self, parent_entropy, parent_size, left_y, right_y):
        w_left, w_right = len(left_y)/parent_size, len(right_y)/parent_size
        left_entropy, right_entropy = self.entropy(left_y), self.entropy(right_y)
        return parent_entropy -(w_left * left_entropy + w_right * right_entropy)
        
    
    # def entropy(self, y): # - More Vesatile
    #     # https://homes.cs.washington.edu/~shapiro/EE596/notes/InfoGain.pdf
    #     _ , counts = np.unique(y, return_counts=True)
    #     prob_features = counts / len(y)
        
    #     return np.sum([-prob * np.log2(prob) for prob in prob_features])
    
    def entropy(self, y): # Only works with integers and float with 0 decimal points, but much more efficient
        hist = np.bincount(y.astype(int)) #should be integers, does not work with floats
        ps = hist / len(y)
        return  -np.sum([p * np.log(p) for p in ps if p > 0])
        
    def fit(self, X, y): 
        self.root = self.build_tree(X, y)
        
    def predict(self, X):
        if self.root == None:
            raise SystemError("Classifier has not been fitted yet!")
        return np.array([self.traverse(x, self.root) for x in X])
    
    def traverse(self, X, node: Node): 
        if node.value != None:
            return node.value
        return self.traverse(X, node.left) if X[node.feature_index] <= node.threshold \
            else self.traverse(X, node.right)
    
    
