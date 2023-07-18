"""Module for K-means model"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.utils.validation import NotFittedError

from Model import UnSupervisedModel
from utils.metrics import euclidean, manhattan


class KMeans(UnSupervisedModel):
    """
    K-means Clustering using lloyd's algorithm

    Paramaters
    ----------

    n_clusters: int
        Number of clusters to form, which also includes the nubmer of centroids to generate

    init: {"k++", "random"}
        Initialization method

        "k++": Perform K-means++ initialization

        "random": Randomly choose the locations for initial centroids

        If provided argument is not on the supported values, "k++" will be used

    n_init:
        Number of times K-Means algorithm is executed. The final results
        is the best output out of the 'n_init' consecutive runs

    max_iter:
        Maximum number of iterations of the K-means algorithm in a single run

    dist_metric: {"euclidean", "manhattan"}
        Metrics to calculate distance between points

        "euclidean": Peform euclidean method. 

        "manhattan": Perform manhttan method

        If provided argument is not on the supported values, "euclidean" will be used

    random_state:
        Value to control the randomness of the model

    tol:
        Relative tolerance with regards to the Frobenius norm of the differences
        in the centroids of two consecutive runs to declare convergence 

    Attributes
    ----------
    All variables in [Parameters]

    init_func:
        Initialization centroid function based on [init] value

    dist_func:
        Distance Function based on [dist_metric] value
   
    labels: ndarray of shape (n_samples,)
        Labels of each point

    centroids: ndarray of shape (n_clusters, n_features)
        Co-ordinates of each centroid

    inertia: float
        Intertia value, ui

    n_iter: int
        Number of iterations run performed

    rng: Generator
        RNG Generator used for randomness 
    """

    def __init__(self,
                 n_clusters: int = 3,
                 *,
                 init = "k++",
                 n_init = 1,
                 max_iter: int = 1,
                 dist_metric: str = "euclidean",
                 tol: float = 1e-4,
                 random_state: Optional[int | np.random.Generator] = None,
                 ):

        self.validate_param(
            n_clusters = n_clusters,
            init = init,
            n_init = n_init,
            max_iter = max_iter,
            dist_metric = dist_metric,
            tol = tol,
            random_state = random_state,
        )

        self.X = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.init = init
        self.INIT_DICT = {"random": self.randomize_centroids, "k++": self.k_plus_centroids}
        self.init_func = self.INIT_DICT[init]
        self.n_init = n_init

        self.dist_metric = dist_metric
        self.DIST_DICT = {"euclidean": euclidean, "manhattan": manhattan}
        self.dist_func = self.DIST_DICT[dist_metric]

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.tol = tol

        self.labels = None
        self.centroids = None
        self.inertia = None

        self.n_iter = 0

    def _fit(self, X: NDArray, plot = False):
        """
        Helper function to perform Kmeans once

        Return:
        n_iter: number of iterations run

        centroids: Co-ordinates of centroids

        labels: Cluster Label of each point

        inertia: WSS value of points based on their cluster
        """
        self.n_features_in = X.shape[1]

        centroids = self.k_plus_centroids()
        n_iter = 0
        cluster_num = np.zeros(X.shape[0]).astype(int)

        while n_iter < self.max_iter:
            old_centroids = centroids.copy()

            ## Assign Cluster
            distances = self.get_distances(X, centroids)
            cluster_num = np.argmin(distances, axis = 1)

            ## Update Centroid
            for i in range(self.n_clusters):
                cluster_row = X[cluster_num == i]
                centroids[i, :] = np.mean(cluster_row, axis = 0) if len(cluster_row) != 0 else centroids[i, :]

            if plot:
                self.plot_clusters(cluster_num, centroids, n_iter)

            ## Check convergence
            if np.linalg.norm(old_centroids - centroids) <= self.tol:
                break

            n_iter += 1

        return n_iter, centroids, cluster_num, self.calc_inertia(X, cluster_num, centroids)

    def fit(self, X: NDArray, plot = False):
        """
        Fit model using k-means clustering from the training dataset ([X], [y])
        """
        super().fit(X)
        self.X = X
        best_inertia = np.inf
        best_fit_return = None

        for _ in range(self.n_init):
            fit_return = self._fit(X, plot)

            if fit_return[3] < best_inertia:
                best_fit_return = fit_return
                best_inertia = fit_return[3]

        self.n_iter, self.centroids, self.labels, self.inertia = best_fit_return

    def _predict(self, x: NDArray):
        """
        Predict class value for instance [X]
        """
        dists = self.get_distances_oneD(x, self.centroids)
        return np.argmin(dists)

    def predict(self, X: NDArray):
        """
        Predict class value for [X]
        """
        if self.X is None:
            raise NotFittedError("KMeans model has not been fitted yet!")
        return np.array([self._predict(x) for x in X])


    ###### Centroid initialization ######


    def init_centroids(self):
        """
        Initialize centroids location based on init parameter given on constructor.

        Same as calling self.init_func
        """
        return self.INIT_DICT[self.init]

    def randomize_centroids(self):
        """
        Initialize centroids location using randomization
        """
        return np.random.uniform(np.amin(self.X, axis = 0),
                                 np.amax(self.X, axis = 0),
                                 size = (self.n_clusters, self.X.shape[1])
                                )

    def k_plus_centroids(self):
        """
        Initialize centroids location using k-means++
        """
        centroids = []
        centroids.append(self.rng.choice(self.X))

        for _ in range(self.n_clusters - 1):
            dists = self.get_distances(self.X, centroids)
            dists = np.amin(dists, axis = 1)

            total_dist = np.sum(dists)
            prob_dists = dists/total_dist

            centroids.append(
                self.rng.choice(self.X, p = prob_dists)
            )

        return np.array(centroids)


    ###### Calculations ######


    def get_distances(self, X: NDArray, centroids: NDArray):
        """
        Return distances between [X] and [centroids]

        Return in the form where row = instances in X, and col = centroids number.
        """
        n = len(centroids)

        dist = np.zeros((X.shape[0], n))
        for i in range(n):
            dist[:, i] = self.dist_func(X, centroids[i], axis = 1)
        return dist

    def get_distances_oneD(self, x: NDArray, centroids: NDArray):
        """
        Return distances between instance [X] and [centroids]

        Return in the form where col = centroids number.
        """
        return np.array([self.dist_func(x, centroids[i]) for i in range(len(centroids))])

    def calc_inertia(self, X: NDArray, labels: NDArray, centroids: NDArray):
        """
        Calculate intertia, i.e, the WSS value
        """
        return np.sum(self.dist_func(X, centroids[labels], axis = 1))


    ###### Plot ######


    def plot_clusters(self, labels: NDArray, centroids: NDArray, iteration: int):
        """
        Plot a 2D kmeans graph after applying PCA on the current [iteration].

        Recommended to only use function on simple datasets with constructor attribute n_init = 1.
        """
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.X)
        centroids_pca = pca.transform(centroids)

        plt.title(f"Iteration: {iteration}")
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c = labels)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    c = range(self.n_clusters), edgecolors="black",
                    marker = "*", s = 200)
        plt.show()


    ###### Validation ######


    def validate_param(self, n_clusters, init, n_init, max_iter, dist_metric, tol, random_state):
        """
        Validate parameter arguments
        """
        if n_clusters < 1:
            raise ValueError(f"n_cluster should be greater than 0. Got a value of {n_clusters} instead.")

        if init not in ("random", "k++"):
            raise ValueError(f"Invalid [init] argument. Supported values are: {('random', 'k++')}")

        if n_init < 1:
            raise ValueError(f"n_init should be greater than 0. Got a value of {n_init} instead.")

        if dist_metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Invalid [dist_metric] argument. Supported values are: {('euclidean', 'manhattan')}")

        if max_iter < 1:
            raise ValueError(f"max_iterations should be greater than 0. Got a value of {max_iter} instead.")

        if tol < 0:
            raise ValueError(f"tol should be a positive number. Got a value of {tol} instead.")

        if isinstance(random_state, int) and random_state < 0:
            raise ValueError(f"random_state integer value should be greater than 0. Got a value of {random_state} instead.")
