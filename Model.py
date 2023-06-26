from abc import ABC

class Model(ABC):
    def predict(self, X):
        raise NotImplementedError
    
    def fit(self, X, y):
        raise NotImplementedError