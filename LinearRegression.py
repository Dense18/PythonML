from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import NotFittedError

from Model import SupervisedModel
from utils.metrics import mse
from utils.utils import batch_samples


class LinearRegression(SupervisedModel):
    """
    Linear regression model using gradient descent
    
    Parameters
    ----------
    fit_intercept:
        Whether to include intercept for this model
    
    Attributes
    ----------
    weights: ndarray of shape(n_feature)
        The coefficient value for each feature variable
    
    bias:
        The intercept value
    """
    
    def __init__(self, *, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        
    def least_squares(self, X, y):
        """
        Fit Linear Regression using Ordinary Least Squares 
        """
        # Formula used: inverse(X'X)X'y
        # Based on https://online.stat.psu.edu/stat462/node/132/
        copy_X = X.copy()
        
        if self.fit_intercept:
            copy_X = np.c_[np.ones(X.shape[0]), copy_X]
        
        inv = np.linalg.inv(np.dot(copy_X.T, copy_X)) # inverse(X'X)
        X_t_y = np.dot(copy_X.T, y) # X'y
        coef = np.dot(inv, X_t_y) # inverse(X'X)X'y
        
        self.bias, self.weights = coef[0], coef[1:]
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fit Linear Regression model from the training dataset ([X], [y])
        """
        super().fit(X, y)
        self.least_squares(X, y)

    def predict(self, X: ArrayLike):
        """
        Predict regression value from [X]
        """
        if self.weights is None:
            raise NotFittedError("Linear Regression model has not beed fitted yet!")
        return np.dot(X, self.weights) + self.bias
    
    
    