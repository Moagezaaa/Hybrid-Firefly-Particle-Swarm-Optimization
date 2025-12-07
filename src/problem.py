"""
Problem model: encoding, decoding, evaluation, constraints, repair.
We use integer encodings:

- placement vector "place_loc": length = num_locations
    place_loc[p] = c+1  -> cloudlet type index (1..T) placed at location p
    place_loc[p] = 0    -> no cloudlet placed at p

- assignment vector "assign_dev": length = num_devices
    assign_dev[e] = p index (0..num_locations-1) of assigned location

Note: A device assigned to p must have place_loc[p] != 0 and distance <= radius of that placed cloudlet.
"""

import numpy as np

class CloudletProblem:
    def __init__(self, instance):
        self.locations = instance['locations']            # (P,2)
        self.devices = instance['devices']                # (E,2)
        self.cloudlet_types = instance['cloudlet_types']  # list
        self.demands = instance['devices_demands']        # list of dicts
        self.placement_cost = instance['placement_cost']  # (T,P)

        self.P = len(self.locations)
        self.E = len(self.devices)
        self.T = len(self.cloudlet_types)

        # Precompute distances device->location
        dev = np.array(self.devices)
        loc = np.array(self.locations)
        self.dist = np.linalg.norm(dev[:, None, :] - loc[None, :, :], axis=2)  # (E,P)

    def decode_solution(self, place_loc, assign_dev):
        """
        No-op here (already in decoded form): returns dict with info
        """
        return {'place_loc': np.array(place_loc, dtype=int), 'assign_dev': np.array(assign_dev, dtype=int)}

    def evaluate(self, place_loc, assign_dev, penalty_coeff=1e6):
        """
        Returns tuple (cost, latency, total_penalty, combined_fitness)
        Combined fitness minimized: latency + lambda * cost + penalty.
        We'll return both objectives and a scalar "fitness" used by the hybrid algorithm.
        """
        place_loc = np.asarray(place_loc, dtype=int)
        assign_dev = np.asarray(assign_dev, dtype=int)

        # placement cost
        placement_total = 0.0
        for p in range(self.P):
            c = place_loc[p]
            if c > 0:
                placement_total += self.placement_cost[c-1, p]  # c stored as 1-based index

        # latency = sum distance of device to assigned location
        lat = 0.0
        for e in range(self.E):
            p = assign_dev[e]
            if p < 0 or p >= self.P:
                lat += 1e6
            else:
                lat += float(self.dist[e, p])

        # Constraint checks and penalty
        penalty = 0.0

        # 1) coverage: device assigned to location must be within coverage radius
        for e in range(self.E):
            p = assign_dev[e]
            if p < 0 or p >= self.P:
                penalty += 1e5
            else:
                c = place_loc[p]
                if c == 0:
                    penalty += 1e5
                else:
                    # cloudlet type c-1
                    radius = self.cloudlet_types[c-1]['R']
                    if self.dist[e, p] > radius + 1e-9:
                        penalty += 1e5

        # 2) capacity: sum of demands of devices assigned to p must be <= cloudlet capacity
        # accumulate per p
        cpu_used = np.zeros(self.P)
        mem_used = np.zeros(self.P)
        sto_used = np.zeros(self.P)
        for e in range(self.E):
            p = assign_dev[e]
            if 0 <= p < self.P:
                d = self.demands[e]
                cpu_used[p] += d['cpu']
                mem_used[p] += d['mem']
                sto_used[p] += d['sto']

        for p in range(self.P):
            c = place_loc[p]
            if c == 0:
                # if resources used but no cloudlet placed -> penalty
                if cpu_used[p] > 0 or mem_used[p] > 0 or sto_used[p] > 0:
                    penalty += 1e6
            else:
                t = self.cloudlet_types[c-1]
                if cpu_used[p] > t['CPU'] + 1e-9:
                    penalty += (cpu_used[p] - t['CPU']) * 1e4
                if mem_used[p] > t['MEM'] + 1e-9:
                    penalty += (mem_used[p] - t['MEM']) * 1e4
                if sto_used[p] > t['STO'] + 1e-9:
                    penalty += (sto_used[p] - t['STO']) * 1e4

        # Combined fitness for scalar optimization
        # weight cost vs latency
        w_cost = 0.4
        w_lat = 0.6
        fitness = w_cost * (placement_total) + w_lat * (lat) + penalty * penalty_coeff

        return {
            'placement_cost': placement_total,
            'latency': lat,
            'penalty': penalty,
            'fitness': fitness
        }

    def random_solution(self, max_active_locations=None):
        """
        Create a feasible-ish random solution:
        - Randomly choose some locations to place cloudlets and a type for each.
        - For each device, assign to nearest location that can host it (w/ radius), else random location.
        """
        if max_active_locations is None:
            max_active_locations = max(1, self.P // 4)

        place_loc = np.zeros(self.P, dtype=int)
        active_count = np.random.randint(1, max_active_locations+1)
        loc_indices = np.random.choice(self.P, size=active_count, replace=False)
        for p in loc_indices:
            # pick cloudlet type randomly
            t = np.random.randint(0, self.T)
            place_loc[p] = t + 1

        # assign each device to nearest active location that covers it, else nearest location
        assign_dev = np.zeros(self.E, dtype=int)
        for e in range(self.E):
            candidates = []
            for p in range(self.P):
                c = place_loc[p]
                if c > 0:
                    radius = self.cloudlet_types[c-1]['R']
                    if self.dist[e, p] <= radius:
                        candidates.append(p)
            if len(candidates) == 0:
                # no covering location -> assign to nearest location (will be penalized)
                assign_dev[e] = int(np.argmin(self.dist[e, :]))
            else:
                assign_dev[e] = int(np.random.choice(candidates))
        return place_loc, assign_dev

    def repair_solution(self, place_loc, assign_dev):
        """
        Attempt to repair easy violations:
        - If a device is assigned to p where place_loc[p]==0 or distance>radius, reassign to nearest feasible p.
        - If capacity violated at a p, try to move some devices out to other feasible locations.
        This is greedy and not guaranteed to fully fix everything, but helps.
        """
        place_loc = place_loc.copy()
        assign_dev = assign_dev.copy()

        # precompute per-device feasible locations (given current placements)
        feasible = [[] for _ in range(self.E)]
        for e in range(self.E):
            for p in range(self.P):
                c = place_loc[p]
                if c > 0:
                    if self.dist[e, p] <= self.cloudlet_types[c-1]['R']:
                        feasible[e].append(p)

        # fix assignments where not feasible
        for e in range(self.E):
            p = assign_dev[e]
            if p < 0 or p >= self.P or (p not in feasible[e]):
                if len(feasible[e]) > 0:
                    assign_dev[e] = int(np.random.choice(feasible[e]))
                else:
                    # try to place a small cloudlet at nearest location to cover it
                    near_p = int(np.argmin(self.dist[e, :]))
                    # pick smallest cloudlet type that can cover (radius)
                    chosen = None
                    for t_idx, t in enumerate(self.cloudlet_types):
                        if self.dist[e, near_p] <= t['R']:
                            chosen = t_idx
                            break
                    if chosen is not None:
                        place_loc[near_p] = chosen + 1
                        assign_dev[e] = near_p
                        feasible[e].append(near_p)
                    else:
                        # leave assignment to nearest location (will be penalized)
                        assign_dev[e] = near_p

        # capacity fix: for overloaded locations, move devices
        def capacity_of(p):
            c = place_loc[p]
            if c == 0:
                return (0,0,0)
            t = self.cloudlet_types[c-1]
            return (t['CPU'], t['MEM'], t['STO'])

        # iterate a few times to try repair
        for _ in range(3):
            cpu_used = np.zeros(self.P)
            mem_used = np.zeros(self.P)
            sto_used = np.zeros(self.P)
            for e in range(self.E):
                p = assign_dev[e]
                if 0 <= p < self.P:
                    d = self.demands[e]
                    cpu_used[p] += d['cpu']
                    mem_used[p] += d['mem']
                    sto_used[p] += d['sto']

            moved = False
            for p in range(self.P):
                c = place_loc[p]
                if c == 0: continue
                cap_cpu, cap_mem, cap_sto = capacity_of(p)
                over_cpu = cpu_used[p] - cap_cpu
                over_mem = mem_used[p] - cap_mem
                over_sto = sto_used[p] - cap_sto
                if over_cpu > 0 or over_mem > 0 or over_sto > 0:
                    # move some devices assigned to p to other feasible locations
                    assigned_devices = [e for e in range(self.E) if assign_dev[e] == p]
                    # sort assigned devices by "heaviest" resource (sum)
                    assigned_devices.sort(key=lambda e: self.demands[e]['cpu']+self.demands[e]['mem']+self.demands[e]['sto'], reverse=True)
                    for e in assigned_devices:
                        # find alternative feasible location for e (other than p)
                        alt = [pp for pp in feasible[e] if pp != p]
                        if len(alt) == 0:
                            continue
                        # choose nearest alt
                        alt.sort(key=lambda pp: self.dist[e, pp])
                        newp = alt[0]
                        # move e
                        d = self.demands[e]
                        cpu_used[p] -= d['cpu']; mem_used[p] -= d['mem']; sto_used[p] -= d['sto']
                        cpu_used[newp] += d['cpu']; mem_used[newp] += d['mem']; sto_used[newp] += d['sto']
                        assign_dev[e] = newp
                        moved = True
                        # update over flags
                        cap_cpu_p = cap_cpu
                        if cpu_used[p] <= cap_cpu and mem_used[p] <= cap_mem and sto_used[p] <= cap_sto:
                            break
            if not moved:
                break

        return place_loc, assign_dev
