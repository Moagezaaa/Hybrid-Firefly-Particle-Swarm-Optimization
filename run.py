"""
Top-level runner:
- generate synthetic instance
- run hybrid FF-PSO
- print results
"""
from data.generate_synthetic import generate_instance
from src.problem import CloudletProblem
from src.experiments import run_simple_experiment
import numpy as np
import json

def pretty_print_sol(sol, metrics, problem):
    print("Placement (location -> cloudlet_type):")
    for p, c in enumerate(sol['place_loc']):
        if c == 0:
            print(f" loc {p}: -")
        else:
            t = problem.cloudlet_types[c-1]
            print(f" loc {p}: type {c-1} (CPU={t['CPU']} MEM={t['MEM']} STO={t['STO']} R={t['R']})")
    print("\nSample device assignments (first 20 devices):")
    for e in range(min(20, problem.E)):
        print(f" device {e} -> loc {sol['assign_dev'][e]} (dist {problem.dist[e, sol['assign_dev'][e]]:.2f})")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

def main():
    inst = generate_instance(num_devices=200, num_locations=20, num_cloudlet_types=6, seed=123)
    prob = CloudletProblem(inst)
    res = run_simple_experiment(prob, pop_size=40, max_iter=200)
    print("Finished. Archive size:", len(res['archive']))
    # print archive solutions summary
    for i, entry in enumerate(res['archive']):
        m = entry['metrics']
        print(f"Archive sol {i}: cost={m['placement_cost']:.2f} latency={m['latency']:.2f} penalty={m['penalty']:.2e}")
    # print best
    best = res['best_solution']
    if best is not None:
        best_metrics = prob.evaluate(best['place_loc'], best['assign_dev'])
        print("\nBest solution metrics:")
        pretty_print_sol(best, best_metrics, prob)

if __name__ == "__main__":
    main()
