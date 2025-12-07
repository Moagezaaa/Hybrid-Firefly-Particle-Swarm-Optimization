import numpy as np
import random

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def one_point_crossover(a, b):
    n = len(a)
    if n < 2:
        return a.copy(), b.copy()
    pt = random.randint(1, n-1)
    c1 = np.concatenate([a[:pt], b[pt:]])
    c2 = np.concatenate([b[:pt], a[pt:]])
    return c1, c2

def uniform_crossover(a, b, prob=0.5):
    n = len(a)
    child1 = a.copy()
    child2 = b.copy()
    for i in range(n):
        if random.random() < prob:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2
