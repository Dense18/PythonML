""" Module for Linear Regression models"""

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import NotFittedError

from Model import SupervisedModel


class LinearRegression(SupervisedModel):
    """
    Linear regression model using Ordinary Least Squares method
    
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
        super().__init__()
        self.fit_intercept = fit_intercept

        self.weights = None
        self.bias = None

    def fit(self, X: NDArray, y: NDArray):
        """
        Fit Linear Regression using Ordinary Least Squares 
        
        Raises
        --------
        ValueError:
            if X or Y have invalid values or shapes
            
        LinAlgError: 
            if a singular matrix is found during fitting process
            
            This could be due to:
            - Linear dependency among the features
            - n rows < nfeatures
        """
        # Formula used: coef = inverse(X'X)X'y
        # Based on https://online.stat.psu.edu/stat462/node/132/
        super().fit(X, y)

        copy_X = X.copy()

        if self.fit_intercept:
            copy_X = np.c_[np.ones(X.shape[0]), copy_X]

        try:
            inv = np.linalg.inv(np.dot(copy_X.T, copy_X)) # inverse(X'X)
        except Exception as err:
            raise np.linalg.LinAlgError(
                   "Found Singularity matrix during fitting process.\nThis could be due to:\n"
                   "  - Linear dependency among the features\n"
                   "  - n_rows < n_features.\n"
                ) from err
        X_t_y = np.dot(copy_X.T, y) # X'y
        coef = np.dot(inv, X_t_y) # inverse(X'X)X'y
        self.bias, self.weights = coef[0], coef[1:]

    def predict(self, X: NDArray):
        """
        Predict regression value from [X]
        """
        if self.weights is None:
            raise NotFittedError("Linear Regression model has not beed fitted yet!")
        return np.dot(X, self.weights) + self.bias
