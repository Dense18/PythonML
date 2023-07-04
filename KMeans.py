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
                 n_clusters: int = 3,
                 *,
                 init = "k++",
                 n_init = 1,
                 max_iterations: int = 1,
                 dist_metric: str = "euclidean",
                 random_seed: Optional[int] = None,
                 tol: float = 1e-4
                 ):
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
        self.init = init
        self.init_dict = {"random": self.randomize_centroids, "k++": self.k_plus_centroids}
        self.init_func = self.init_dict.get(init, "k++")
        self.n_init = n_init
        
        self.dist_metric = dist_metric
        self.dist_dict = {"euclidean": distUtil.euclidean, "manhattan": distUtil.manhattan}
        self.dist_func = self.dist_dict[dist_metric]
        
        self.rng = np.random.default_rng(random_seed)
        
        self.tol = tol
        
        self.labels = None
        self.centroids = None
        self.inertia = None
        
        self.n_iter = 0
    
    # def fit(self, X: ArrayLike, plot = False):
    #     """
    #     Fits the model based on the training set [X] 
    #     """
    #     best_n_iter = None
    #     best_centroids = None
    #     best_labels = None
    #     best_interia = np.inf
        
    #     best_fit_return = None
        
    #     for _ in range(self.n_init):
    #         n_iter, centroids, labels, intertia = self._fit(X, plot)
    #         fit_return = self._fit(X, plot)
    #         if intertia < best_interia:
    #             best_n_iter = n_iter
    #             best_centroids = centroids
    #             best_labels = labels
    #             best_interia = intertia
        
    #     self.n_iter = best_n_iter
    #     self.centroids = best_centroids
    #     self.labels = best_labels
    #     self.inertia = best_interia
        
    def fit(self, X: ArrayLike, plot = False):
        """
        Fits the model based on the training set [X] 
        """
        best_inertia = np.inf        
        best_fit_return = None
        
        for _ in range(self.n_init):
            fit_return = self._fit(X, plot)
            
            if fit_return[3] < best_inertia:
                best_fit_return = fit_return
                best_inertia = fit_return[3]
        
        self.n_iter, self.centroids, self.labels, self.inertia = best_fit_return
    
    def _fit(self, X: ArrayLike, plot = False):
        """
        Helper function to perform Kmeans once
        
        Returns:
        n_iter: number of iterations run
        
        centroids: Co-ordinates of centroids
        
        labels: Cluster Label of each point
        
        inertia: WSS value of points based on their cluster
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = X
        
        centroids = self.k_plus_centroids()
        n_iter = 0
        cluster_num = np.zeros(X.shape[0]).astype(int)
        
        while n_iter < self.max_iterations:
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
            if np.max(old_centroids - centroids) <= self.tol:
                break
            
            n_iter += 1
        
        return n_iter, centroids, cluster_num, self.calc_inertia(X, cluster_num, centroids)
    
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
        Returns distances between [X] and [centroids]
        
        Returns in the form where row = instances in X, and col = centroids number.
        """
        n = len(centroids)
        
        dist = np.zeros((X.shape[0], n))
        for i in range(n):
            dist[:, i] = self.dist_func(X, centroids[i], axis = 1)
        return dist
    
    def get_distances_oneD(self, x: ArrayLike, centroids: ArrayLike):
        """
        Returns distances between instance [X] and [centroids]
        
        Returns in the form where col = centroids number.
        """
        return np.array([self.dist_func(x, centroids[i]) for i in range(len(centroids))])
    
    def plot_clusters(self, labels: ArrayLike, centroids: ArrayLike, iteration: int):
        """
        Plots a 2D kmeans graph after applying PCA on the current [iteration].
        
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
    
    def calc_inertia(self, X: ArrayLike, labels: ArrayLike, centroids: ArrayLike):
        """
        Calculates intertia, i.e, the WSS value
        """
        return np.sum(self.dist_func(X, centroids[labels], axis = 1))
    
    def _predict(self, x: ArrayLike):
        """
        Predicts class value for instance [X]
        """
        dists = self.get_distances_oneD(x, self.centroids)
        return np.argmin(dists)
        
    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        if self.X is None:
            raise NotFittedError("KMeans model has not been fitted yet!")
        return np.array([self._predict(x) for x in X])