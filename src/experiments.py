from src.hybrid_ff_pso import HybridFFPSO
import numpy as np

def run_simple_experiment(problem, pop_size=40, max_iter=200):
    algo = HybridFFPSO(problem, pop_size=pop_size, max_iter=max_iter)
    res = algo.run(verbose=True)
    return res
