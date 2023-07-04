from _typeshed import SupportsWrite
from abc import ABCMeta
import numpy as np
from numpy.typing import ArrayLike
from Model import UnSupervisedModel
import utils.distUtil as distUtil

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from typing import Optional

from sklearn.utils.validation import NotFittedError

class KMeans(UnSupervisedModel):
    """
    K-means Clustering using lloyd's algorithm
    """
    def __init__(self, 
                 init = "k++",
                 n_clusters: int = 3,
                 max_iterations: int = 1,
                 dist_metric: str = "euclidean",
                 random_seed: Optional[int] = None,
                 tol: float = 1e-4) -> None:
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
        self.init = init
        self.init_dict = {"random": self.randomize_centroids, "k++": self.k_plus_centroids}
        self.init_func = self.init_dict.get(init, "k++")
        
        self.dist_metric = dist_metric
        self.dist_dict = {"euclidean": distUtil.euclidean, "manhattan": distUtil.manhattan}
        self.dist_func = self.dist_dict[dist_metric]
        
        self.rng = np.random.default_rng(random_seed)
        
        self.tol = tol
        
        self.labels = None
        self.centroids = None
        
        self.iter_count = 0
        
    def fit(self, X: ArrayLike, plot = False):
        """
        Fits the model based on the training set [X]
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = X
        
        centroids = self.k_plus_centroids()
        iter_count = 0
        cluster_num = np.zeros(X.shape[0]).astype(int)
        
        while iter_count < self.max_iterations:
            old_centroids = centroids.copy()
            
            ## Assign Cluster
            distances = self.get_distances(X, centroids)
            cluster_num = np.argmin(distances, axis = 1)
            
            ## Update Centroid based on the mean of the the points on each cluster
            for i in range(self.n_clusters):
                cluster_row = X[cluster_num == i] 
                centroids[i, :] = np.mean(cluster_row, axis = 0) if len(cluster_row) != 0 else centroids[i, :]
            
            if plot:
                self.plot_clusters(cluster_num, centroids, iter_count)
            
            ## Check convergence
            if np.max(old_centroids - centroids) <= self.tol:
                break
            
            iter_count += 1
        
        self.iter_count = iter_count
        self.centroids = centroids
        self.labels = cluster_num
    
    def init_centroids(self):
        """
        Initialize centroids location based on init parameter given on constructor.
        
        Same as calling self.init_func
        """
        return self.init_dict[self.init]
    
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
        Initialized centroids location using k-means++
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
        
    
    def get_distances(self, X: ArrayLike, centroids: ArrayLike):
        """
        Returns distances between [X] and [centroids], where distance metric is determined in the constructor
        
        Returns in the form where row = instances in X, and col = centroids number.
        """
        n = len(centroids)
        dist = np.zeros((X.shape[0], n))
        for i in range(n):
            dist[:, i] = self.dist_func(X, centroids[i], axis = 1)
        return dist
    
    def plot_clusters(self, labels: ArrayLike, centroids: ArrayLike, iteration: int):
        """
        Plots a 2D kmeans graph after applying PCA on the current [iteration].
        
        Recommended to only use function on simple datasets.
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
    
    def _predict(self, x: ArrayLike):
        """
        Predicts class value for instance [X]
        """
        if self.centroids == None:
            raise NotFittedError("KMeans model has not been fitted yet!")
        dists = self.get_distances(x, self.centroids)
        
    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        return np.array([self._predict(x) for x in X])