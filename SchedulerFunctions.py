import numpy as np

"""
    Collection of Learning rate scheduler functions
"""
def constant_decay():
    return lambda epoch, init_lr: init_lr

def time_decay(delay: float):
    def inner_func(epoch, init_lr):
        return init_lr / (1 + delay * epoch)
    return inner_func

def exp_decay(exp_k: float):
    def inner_func(epoch, init_lr):
        return init_lr * np.exp(-exp_k * epoch)
    return inner_func
    