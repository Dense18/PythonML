""" Module for Stochastic Gradient Descent models"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import NotFittedError

from Model import SupervisedModel
from utils.metrics import mse
from utils.utils import batch_samples


class SGDRegression(SupervisedModel):
    """
    Linear regression model using stochastic gradient descent

    Parameters
    ----------
    init_learning:
        Learning rate for gradient descent

    scheduler_func:
        Scheduler callable function to update the learning rate
 
        Function should have the form (epoch: int, initial_learning_rate: float) -> current_learning_rate: float

    decay:
        Decay parameter for the default learning rate schedule (see notes below)

    algo: {gd, sgd}
        Algorithm to fit the Linear Regression Method

        gd: Gradient Descent

        sgd: Stochastic gradient descent

    tol:
        Relative tolerance with regards to the differences of 
        the gradient descent in two consecutive run to declare convergence

    batch_size:
        Number of samples used per batch in Stochastic gradient descent
  
    max_iters:
        Maximum numer of iterations to run gradient descent

    random_state:
            Value to control the randomness of the model

    Attributes
    ----------
    weights: ndarray of shape(n_feature)
        The coefficient value for each feature variable

    bias:
        The intercept value

    n_iter:
        Number of iterations run performed in gradient descent

    rng:
        RNG Generator used for randomness 
  

    Notes
    --------
    Default learning scheduler function is time decay function:
        learning_rate = initial_learning rate / (1 + decay + epoch)
    """

    def __init__(self,
                 *,
                 algo: str = "sgd",
                 init_learning: float = 0.01,
                 scheduler_func: Callable[[int], float] = None,
                 decay: float = 0,
                 tol: float = 1e-4,
                 batch_size: int = 64,
                 max_iters: int = 100,
                 random_state: Optional[int | np.random.Generator] = None,
                 ):
        super().__init__()
        self.validate_params(
            algo = algo,
            init_learning = init_learning,
            scheduler_func = scheduler_func,
            decay = decay,
            tol = tol,
            batch_size = batch_size,
            max_iters = max_iters,
            random_state = random_state
        )

        self.algo = algo

        self.init_learning = init_learning
        self.tol = tol
        self.batch_size = batch_size
        self.max_iters = max_iters

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.scheduler_func = self.time_decay if scheduler_func is None else scheduler_func
        self.decay = decay

        self.weights = None
        self.bias = 0
        self.n_iter = 0

    def gradient_descent(self, X: NDArray, y: NDArray):
        """
        Fit Gradient Descent Regression from the training dataset ([X], [y])
        """
        n_features = X.shape[1]
        weights, bias = np.zeros(n_features), 0

        X_batch, y_batch = batch_samples(X, y, self.batch_size, random_state = self.random_state) \
            if self.algo == "sgd" else (X, y)
        y_pred = np.dot(X_batch, weights) + bias

        loss = mse(y_batch, y_pred)
        n_iter = 0
        learning_rate = self.init_learning

        while n_iter < self.max_iters:
            old_loss = loss

            ## Obtain random batch
            X_batch, y_batch = batch_samples(X, y, self.batch_size, random_state = self.random_state) \
                if self.algo == "sgd" else (X, y)
            n_batch_samples = len(X_batch)

            dldw = -2/n_batch_samples * np.dot(X_batch.T, y_batch - y_pred)
            dldb = -2/n_batch_samples * np.sum(y_batch - y_pred)

            weights = weights - learning_rate * dldw
            bias = bias - learning_rate * dldb

            y_pred = np.dot(X_batch, weights) + bias
            loss = mse(y_batch, y_pred)

            n_iter += 1
            learning_rate = self.scheduler_func(n_iter, self.init_learning)

            # Check for Convergence
            if np.linalg.norm(loss - old_loss) < self.tol:
                print(f"Converges at iteration {n_iter}")
                break

        self.n_iter = n_iter
        self.weights = weights
        self.bias = bias

    def time_decay(self, epoch: int, init_learning: float):
        return init_learning / (1 + self.decay * epoch)

    def fit(self, X: NDArray, y: NDArray):
        """
        Fit model using gradient descent from the training dataset ([X], y)
        """
        super().fit(X, y)
        self.gradient_descent(X, y)

    def predict(self, X: NDArray):
        """
        Predict regression value for [X]
        """
        if self.weights is None:
            raise NotFittedError("Linear Regression model has not beed fitted yet!")

        return np.dot(X, self.weights) + self.bias


    ####### Validation #######


    def validate_params(self,
                 algo,
                 init_learning,
                 scheduler_func,
                 decay,
                 tol,
                 batch_size,
                 max_iters,
                 random_state
                 ):

        if algo not in ("gd", "sdg"):
            raise ValueError(f"Invalid [algo] value. Suppored Values are: 'gd', 'sgd'")

        if tol < 0:
            raise ValueError(f"tol should be a positive number. Got a value of {tol} instead.")

        if batch_size <= 0:
            raise ValueError(f"batch size should be greater than 0. Got a value of {batch_size} instead.")

        if max_iters < 0:
            raise ValueError(f"max_iters should be a positive number. Got a value of {max_iters} instead.")

        if isinstance(random_state, int) and random_state < 0:
            raise ValueError(f"random_state integer value should be greater than 0. Got a value of {random_state} instead.")
