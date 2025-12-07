"""
Hybrid Firefly + Discrete PSO for Cloudlet placement.

Representation:
- place_loc: int array length P (0 no cloudlet, otherwise 1..T)
- assign_dev: int array length E (0..P-1) device assigned location index

Population of particles each with:
- current solution (place_loc, assign_dev)
- particle velocity (for discrete, we store probabilities or swap tendencies)
- personal best (pbest) and its fitness
- firefly brightness = inverse of fitness (lower fitness => brighter)

Algorithm outline:
- Initialize population randomly
- Repeat:
    - Evaluate fitness
    - Non-dominated sorting not used here; we keep scalar fitness to guide moves
    - For each particle:
        - PSO update: move towards personal best and global best using discrete operators
        - Firefly step: move towards brighter fireflies by mixing assignments & placements
        - Mutation: random changes
        - Repair
- Keep track of best found (Pareto front approximated by storing nondominated seen)
"""

import numpy as np
import random
from copy import deepcopy
from tqdm import trange

from src.utils import one_point_crossover, uniform_crossover

class Particle:
    def __init__(self, place_loc, assign_dev):
        self.place_loc = place_loc.copy()
        self.assign_dev = assign_dev.copy()
        # velocities: for discrete representation keep probability matrices
        self.v_place = np.zeros_like(place_loc, dtype=float)  # not heavily used here
        self.v_assign = np.zeros_like(assign_dev, dtype=float)
        self.pbest_place = place_loc.copy()
        self.pbest_assign = assign_dev.copy()
        self.pbest_score = np.inf

class HybridFFPSO:
    def __init__(self, problem, pop_size=40, max_iter=300, seed=42):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        random.seed(seed)
        np.random.seed(seed)

        self.population = []
        for _ in range(pop_size):
            p_loc, a_dev = problem.random_solution()
            part = Particle(p_loc, a_dev)
            self.population.append(part)

        self.global_best = None
        self.global_best_score = np.inf
        self.archive = []  # store nondominated solutions (simple list of dicts)

    def evaluate_population(self):
        evals = []
        for part in self.population:
            r = self.problem.evaluate(part.place_loc, part.assign_dev)
            evals.append(r)
        return evals

    def nondominated_archive_update(self, sol, metrics):
        """
        Keep a simple archive of nondominated solutions (w.r.t two objectives: cost & latency).
        metrics: dict with 'placement_cost','latency','penalty','fitness'
        """
        # if dominated by any in archive -> ignore
        dominated = False
        removals = []
        for i,entry in enumerate(self.archive):
            m = entry['metrics']
            # entry dominates sol?
            if (m['placement_cost'] <= metrics['placement_cost'] and
                m['latency'] <= metrics['latency'] and
                (m['placement_cost'] < metrics['placement_cost'] or m['latency'] < metrics['latency'])):
                dominated = True
                break
            # sol dominates entry?
            if (metrics['placement_cost'] <= m['placement_cost'] and
                metrics['latency'] <= m['latency'] and
                (metrics['placement_cost'] < m['placement_cost'] or metrics['latency'] < m['latency'])):
                removals.append(i)
        if not dominated:
            # remove dominated entries
            for idx in sorted(removals, reverse=True):
                self.archive.pop(idx)
            self.archive.append({'place_loc': sol['place_loc'].copy(), 'assign_dev': sol['assign_dev'].copy(), 'metrics': metrics.copy()})

    def local_pso_move(self, part, w=0.7, c1=1.2, c2=1.2):
        """
        Discrete PSO-like move:
        - For placements: with probability based on difference to pbest/gbest, switch types or toggle placement.
        - For assignments: with probability move towards pbest and gbest assigned location.
        """
        # combine pbest and global best if exists
        g_place = self.global_best['place_loc'] if self.global_best is not None else part.pbest_place
        g_assign = self.global_best['assign_dev'] if self.global_best is not None else part.pbest_assign

        P = self.problem.P
        E = self.problem.E

        # Update place_loc: for each location possibly adopt pbest/gbest choice
        for p in range(P):
            if random.random() < 0.3:
                part.place_loc[p] = part.pbest_place[p]
            if random.random() < 0.2:
                part.place_loc[p] = g_place[p]

            # small chance to randomly toggle placement or change type
            if random.random() < 0.05:
                if part.place_loc[p] == 0:
                    part.place_loc[p] = random.randint(1, self.problem.T)
                else:
                    if random.random() < 0.5:
                        part.place_loc[p] = 0
                    else:
                        part.place_loc[p] = random.randint(1, self.problem.T)

        # Update assignment: for each device move to pbest/gbest with some probability
        for e in range(E):
            if random.random() < 0.35:
                part.assign_dev[e] = part.pbest_assign[e]
            if random.random() < 0.25:
                part.assign_dev[e] = g_assign[e]
            # random jump
            if random.random() < 0.02:
                part.assign_dev[e] = random.randint(0, self.problem.P-1)

    def firefly_move(self, part, brighter_part, beta0=0.8, gamma=1.0):
        """
        Move this particle towards a brighter particle by mixing placements and assignments.
        The intensity of move depends on difference between solutions (hamming).
        """
        P = self.problem.P
        E = self.problem.E

        # placements: adopt some positions and types from brighter
        for p in range(P):
            if brighter_part.place_loc[p] != part.place_loc[p]:
                if random.random() < 0.5:
                    part.place_loc[p] = brighter_part.place_loc[p]

        # assignments: adopt some assignments
        for e in range(E):
            if brighter_part.assign_dev[e] != part.assign_dev[e]:
                # probability proportional to difference
                if random.random() < 0.4:
                    part.assign_dev[e] = brighter_part.assign_dev[e]

    def mutate(self, part, pm_place=0.02, pm_assign=0.03):
        # placements random mutation
        for p in range(self.problem.P):
            if random.random() < pm_place:
                if part.place_loc[p] == 0:
                    part.place_loc[p] = random.randint(1, self.problem.T)
                else:
                    if random.random() < 0.5:
                        part.place_loc[p] = 0
                    else:
                        part.place_loc[p] = random.randint(1, self.problem.T)
        # assignments mutation
        for e in range(self.problem.E):
            if random.random() < pm_assign:
                part.assign_dev[e] = random.randint(0, self.problem.P-1)

    def run(self, verbose=True):
        best_history = []
        for it in trange(self.max_iter, desc="Iter"):
            # Evaluate population
            evals = self.evaluate_population()
            # update personal best and global best
            for i, part in enumerate(self.population):
                metrics = evals[i]
                if metrics['fitness'] < part.pbest_score:
                    part.pbest_score = metrics['fitness']
                    part.pbest_place = part.place_loc.copy()
                    part.pbest_assign = part.assign_dev.copy()
                # update global
                if metrics['fitness'] < self.global_best_score:
                    self.global_best_score = metrics['fitness']
                    self.global_best = {'place_loc': part.place_loc.copy(), 'assign_dev': part.assign_dev.copy()}
                # update archive nondominated
                sol = {'place_loc': part.place_loc, 'assign_dev': part.assign_dev}
                self.nondominated_archive_update(sol, metrics)

            # Sort by brightness (fitness)
            sorted_idx = sorted(range(len(self.population)), key=lambda i: evals[i]['fitness'])
            # Firefly interactions: for each (i) try to move towards brighter (j) with j<i
            for ii in range(len(self.population)):
                i = sorted_idx[ii]
                for jj in range(ii):
                    j = sorted_idx[jj]
                    # if j is brighter (lower fitness)
                    if evals[j]['fitness'] < evals[i]['fitness']:
                        self.firefly_move(self.population[i], self.population[j])

            # PSO moves + mutation + repair
            for part in self.population:
                self.local_pso_move(part)
                self.mutate(part)
                # repair
                part.place_loc, part.assign_dev = self.problem.repair_solution(part.place_loc, part.assign_dev)

            # record best metrics
            best_history.append(self.global_best_score)
            if verbose and (it % max(1, self.max_iter//10) == 0):
                print(f"Iter {it:4d}: best fitness {self.global_best_score:.4e}, archive size {len(self.archive)}")

        return {
            'archive': self.archive,
            'best_score': self.global_best_score,
            'best_solution': self.global_best,
            'history': best_history
        }
