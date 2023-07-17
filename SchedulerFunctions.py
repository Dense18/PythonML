"""
    Module consisting of Learning rate scheduler functions used for Gradient Descent
"""
import numpy as np


def constant_decay():
    """
    Constant decay learning rate
    """
    return lambda epoch, init_lr: init_lr

def time_decay(delay: float):
    """
    Time decay learning rate
    """
    def inner_func(epoch, init_lr):
        return init_lr / (1 + delay * epoch)
    return inner_func

def exp_decay(exp_k: float):
    """
    Expotential decay learning rate
    """
    def inner_func(epoch, init_lr):
        return init_lr * np.exp(-exp_k * epoch)
    return inner_func
