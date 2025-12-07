"""
Generate a small synthetic instance for testing.
Saves nothing to disk; returns structures to caller.
"""
import numpy as np
import random

def generate_instance(num_devices=200, num_locations=20, num_cloudlet_types=5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Candidate points (locations) coordinates
    locations = np.random.rand(num_locations, 2) * 100.0  # in km units

    # Devices coordinates
    devices = np.random.rand(num_devices, 2) * 100.0

    # Cloudlet types: capacity and radius and cost
    cloudlet_types = []
    for i in range(num_cloudlet_types):
        cpu = random.choice([10, 20, 40, 80])  # GHz
        mem = random.choice([16, 32, 64, 128])  # GB
        sto = random.choice([200, 500, 1000])   # GB
        radius = random.choice([10, 20, 30])    # coverage radius
        base_cost = 1000 + cpu * 10 + mem * 5 + sto * 0.1 + radius * 20
        cloudlet_types.append({
            'id': i,
            'CPU': cpu,
            'MEM': mem,
            'STO': sto,
            'R': radius,
            'base_cost': base_cost
        })

    # Device demands
    devices_demands = []
    for i in range(num_devices):
        cpu = random.choice([1,2,4])
        mem = random.choice([1,2,4])
        sto = random.choice([1,5,10])
        devices_demands.append({'cpu': cpu, 'mem': mem, 'sto': sto})

    # Cost of placing each cloudlet type at each candidate location (add small location variation)
    placement_cost = np.zeros((num_cloudlet_types, num_locations))
    for c in range(num_cloudlet_types):
        for p in range(num_locations):
            # add some location-based factor
            placement_cost[c, p] = cloudlet_types[c]['base_cost'] * (1 + 0.1 * np.random.rand())

    return {
        'locations': locations,
        'devices': devices,
        'cloudlet_types': cloudlet_types,
        'devices_demands': devices_demands,
        'placement_cost': placement_cost
    }

if __name__ == "__main__":
    inst = generate_instance()
    print("Generated instance: locations", inst['locations'].shape, "devices", len(inst['devices']))
