import functools
import logging
import time
from collections import Counter

import numpy as np
from scipy import stats as st

logger = logging.getLogger("mytest")

def with_logging(func):
    @functools.wraps(func)
    def inner_func(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        output = func(*args, **kwargs)
        logger.info(f"Finished calling {func.__name__}")
        return output
    return inner_func


def timer(func):
    @functools.wraps(func)
    def inner_func(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        time_taken = time.time() - start
        print(f"Time taken: {time_taken}")
    return inner_func


def timer_repeat(repeat):
    def decorator(func):
        @functools.wraps(func)
        def inner_func(*args, **kwargs):
            start = time.time()
            for _ in range(repeat):
                func(*args, **kwargs)
            time_taken = time.time() - start
            logger.info(f"Time taken for {repeat} operations: {time_taken}")
        return inner_func
    return decorator
    

def timer_repeat_by_calling_timer(repeat):
    def decorator(func):#func = add
        def inside(*args, **kwargs):
            for _ in range(repeat):
                timer(func)(*args, **kwargs)
            logger.info(f"Executed {repeat} times")
        return inside
    
    return decorator
    
def timer_for_partial(func, repeat: int):
    @functools.wraps(func)
    def inner_func(*args, **kwargs):
        start = time.time()
        for _ in range(repeat):
            func(*args, **kwargs)
        time_taken = time.time() - start
        logger.info(f"Time taken for {repeat} operations: {time_taken}")
    return inner_func

timer_with_default = functools.partial(timer_for_partial, repeat = 1)


@with_logging
@timer_with_default
def add(x, y):
    return x + y

def main():
    logging.basicConfig(level=logging.INFO)
    value = add(1,2)
    # for i in range(12):
    #     add(1,2)
    
    
if __name__ == "__main__":
    main()

# x_column = np.random.randint(1, 10000, (1000))