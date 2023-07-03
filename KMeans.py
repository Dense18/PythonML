import numpy as np
from numpy.typing import ArrayLike
from Model import UnSupervisedModel
from utils.distUtil import euclidean

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KMeans(UnSupervisedModel):
    def __init__(self, 
                 n_clusters: int = 3,
                 max_iterations: int = 1) -> None:
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
        self.labels = None
        self.centroids = None
    
    def fit(self, X: ArrayLike, plot = False):
        """
        Fits the model based on the training set [X]
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = X
        
        centroids = np.random.uniform(np.amin(X, axis = 0), 
                                      np.amax(X, axis = 0), 
                                      size = (self.n_clusters, X.shape[1])
                                      )
        iter_count = 0
        cluster_num = np.zeros(X.shape[0]).astype(int)
        
        while iter_count < self.max_iterations:
            old_cluster = cluster_num.copy()
            
            ## Assign Cluster
            distances = self.get_distances(X, centroids)
            cluster_num = np.argmin(distances, axis = 1)
            
            ## Update Centroid based on the mean of the the points on each cluster
            for i in range(self.n_clusters):
                centroids[i, :] = np.mean(X[cluster_num == i], axis =0)
            
            if plot:
                self.plot_clusters(cluster_num, centroids, iter_count)
            
            ## Check convergence
            if all(old_cluster == cluster_num):
                break
            iter_count += 1
        
        self.centroids = centroids
        self.labels = cluster_num
    
    def get_distances(self, X, centroids):
        """
        Returns euclidean distances between [X] and centroids
        """
        dist = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            dist[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        return dist
    
    def plot_clusters(self, labels: ArrayLike, centroids: ArrayLike, iteration: int):
        """
        Plots a 2D kmeans graph after applying PCA on the current [iteration]
        """
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.X)
        centroids_pca = pca.transform(centroids)
        
        plt.figure()
        plt.title(f"Iteration: {iteration}")
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c = labels)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    c = range(self.n_clusters), edgecolors="black", 
                    marker = "*", s = 200)
        plt.show()
    
    def predict(self, X: ArrayLike):
        """
        Predicts class value for [X]
        """
        #TODO:
        pass