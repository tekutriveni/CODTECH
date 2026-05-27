"""
Genetic Algorithms for Optimization
=====================================
3 Problems solved using GA:
  1. Function Optimization     - Find max/min of a mathematical function
  2. Knapsack Problem          - Best items to pack within weight limit
  3. Travelling Salesman (TSP) - Shortest route visiting all cities
"""

import numpy as np
import random
import json
import time

random.seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════
# PROBLEM 1: FUNCTION OPTIMIZATION
# Maximize f(x) = sin(x) * cos(x/2) + x/10  in range [0, 20]
# ════════════════════════════════════════════════════════════

class FunctionOptimizationGA:
    def __init__(self, pop_size=100, generations=200, mutation_rate=0.01, crossover_rate=0.8):
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate= crossover_rate
        self.chromosome_len= 16          # 16-bit binary → range [0, 20]
        self.x_min, self.x_max = 0, 20

    # ── Encoding / Decoding ──────────────────────────────────
    def decode(self, chromosome):
        decimal = int("".join(map(str, chromosome)), 2)
        return self.x_min + decimal * (self.x_max - self.x_min) / (2**self.chromosome_len - 1)

    # ── Fitness ──────────────────────────────────────────────
    def fitness(self, chromosome):
        x = self.decode(chromosome)
        return np.sin(x) * np.cos(x / 2) + x / 10   # objective function

    # ── Selection (Tournament) ───────────────────────────────
    def tournament_select(self, population, fitnesses, k=3):
        idx = random.sample(range(len(population)), k)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best]

    # ── Crossover (Single-Point) ─────────────────────────────
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.chromosome_len - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1[:], parent2[:]

    # ── Mutation (Bit Flip) ──────────────────────────────────
    def mutate(self, chromosome):
        return [bit ^ 1 if random.random() < self.mutation_rate else bit
                for bit in chromosome]

    # ── Run ──────────────────────────────────────────────────
    def run(self):
        population = [[random.randint(0, 1) for _ in range(self.chromosome_len)]
                      for _ in range(self.pop_size)]

        history = []
        best_overall, best_fitness = None, float('-inf')

        for gen in range(self.generations):
            fitnesses = [self.fitness(c) for c in population]
            best_idx  = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_fitness:
                best_fitness  = fitnesses[best_idx]
                best_overall  = population[best_idx][:]

            history.append({
                'generation': gen,
                'best_fitness': round(best_fitness, 6),
                'avg_fitness':  round(float(np.mean(fitnesses)), 6)
            })

            # New generation
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            population = new_pop[:self.pop_size]

        best_x = self.decode(best_overall)
        return {
            'problem': 'Function Optimization',
            'best_x':  round(best_x, 6),
            'best_fitness': round(best_fitness, 6),
            'formula': 'f(x) = sin(x)*cos(x/2) + x/10',
            'history': history
        }


# ════════════════════════════════════════════════════════════
# PROBLEM 2: 0/1 KNAPSACK PROBLEM
# Maximize value of items packed within weight capacity
# ════════════════════════════════════════════════════════════

class KnapsackGA:
    ITEMS = [
        {'name': 'Laptop',     'weight': 3,  'value': 150},
        {'name': 'Phone',      'weight': 1,  'value': 80},
        {'name': 'Tablet',     'weight': 2,  'value': 90},
        {'name': 'Camera',     'weight': 2,  'value': 100},
        {'name': 'Headphones', 'weight': 1,  'value': 60},
        {'name': 'Book',       'weight': 1,  'value': 20},
        {'name': 'Charger',    'weight': 1,  'value': 40},
        {'name': 'Clothes',    'weight': 4,  'value': 30},
        {'name': 'Shoes',      'weight': 3,  'value': 25},
        {'name': 'Watch',      'weight': 1,  'value': 70},
    ]
    CAPACITY = 10

    def __init__(self, pop_size=50, generations=150, mutation_rate=0.02):
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.n_items       = len(self.ITEMS)

    def fitness(self, chromosome):
        total_weight = sum(chromosome[i] * self.ITEMS[i]['weight'] for i in range(self.n_items))
        total_value  = sum(chromosome[i] * self.ITEMS[i]['value']  for i in range(self.n_items))
        return total_value if total_weight <= self.CAPACITY else 0

    def tournament_select(self, population, fitnesses):
        idx  = random.sample(range(len(population)), 3)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best]

    def crossover(self, p1, p2):
        point = random.randint(1, self.n_items - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def mutate(self, chromosome):
        return [bit ^ 1 if random.random() < self.mutation_rate else bit
                for bit in chromosome]

    def run(self):
        population = [[random.randint(0, 1) for _ in range(self.n_items)]
                      for _ in range(self.pop_size)]

        history, best_overall, best_val = [], None, 0

        for gen in range(self.generations):
            fitnesses = [self.fitness(c) for c in population]
            best_idx  = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_val:
                best_val     = fitnesses[best_idx]
                best_overall = population[best_idx][:]

            history.append({'generation': gen, 'best_value': best_val,
                            'avg_value': round(float(np.mean(fitnesses)), 2)})

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            population = new_pop[:self.pop_size]

        packed = [self.ITEMS[i] for i in range(self.n_items) if best_overall[i] == 1]
        total_w = sum(item['weight'] for item in packed)
        return {
            'problem': 'Knapsack',
            'packed_items': [i['name'] for i in packed],
            'total_value':  best_val,
            'total_weight': total_w,
            'capacity':     self.CAPACITY,
            'history':      history
        }


# ════════════════════════════════════════════════════════════
# PROBLEM 3: TRAVELLING SALESMAN PROBLEM (TSP)
# Find shortest route visiting 10 cities exactly once
# ════════════════════════════════════════════════════════════

class TSPGA:
    CITIES = {
        'Hyderabad': (17.38, 78.47), 'Vijayawada':  (16.51, 80.62),
        'Chennai':   (13.08, 80.27), 'Bangalore':   (12.97, 77.59),
        'Mumbai':    (19.07, 72.87), 'Pune':        (18.52, 73.86),
        'Delhi':     (28.67, 77.22), 'Kolkata':     (22.57, 88.36),
        'Jaipur':    (26.91, 75.79), 'Ahmedabad':   (23.02, 72.57),
    }

    def __init__(self, pop_size=100, generations=500, mutation_rate=0.02):
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.city_names    = list(self.CITIES.keys())
        self.n_cities      = len(self.city_names)
        self._build_dist_matrix()

    def _build_dist_matrix(self):
        n = self.n_cities
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                lat1, lon1 = self.CITIES[self.city_names[i]]
                lat2, lon2 = self.CITIES[self.city_names[j]]
                self.dist[i][j] = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 111  # ~km

    def route_distance(self, route):
        return sum(self.dist[route[i]][route[(i+1) % self.n_cities]]
                   for i in range(self.n_cities))

    def fitness(self, route):
        return 1 / self.route_distance(route)

    def tournament_select(self, population, fitnesses):
        idx  = random.sample(range(len(population)), 5)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best]

    # Order Crossover (OX)
    def crossover(self, p1, p2):
        a, b = sorted(random.sample(range(self.n_cities), 2))
        child = [-1] * self.n_cities
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(self.n_cities):
            if child[i] == -1:
                child[i] = fill[idx]; idx += 1
        return child

    # Swap mutation
    def mutate(self, route):
        route = route[:]
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n_cities), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        population = [random.sample(range(self.n_cities), self.n_cities)
                      for _ in range(self.pop_size)]

        history, best_route, best_dist = [], None, float('inf')

        for gen in range(self.generations):
            fitnesses = [self.fitness(r) for r in population]
            dists     = [self.route_distance(r) for r in population]
            best_idx  = np.argmin(dists)

            if dists[best_idx] < best_dist:
                best_dist  = dists[best_idx]
                best_route = population[best_idx][:]

            history.append({'generation': gen,
                            'best_distance': round(best_dist, 2),
                            'avg_distance':  round(float(np.mean(dists)), 2)})

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                c1 = self.crossover(p1, p2)
                c2 = self.crossover(p2, p1)
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            population = new_pop[:self.pop_size]

        route_names = [self.city_names[i] for i in best_route] + [self.city_names[best_route[0]]]
        return {
            'problem': 'TSP',
            'best_route': route_names,
            'best_distance_km': round(best_dist, 2),
            'history': history
        }


# ════════════════════════════════════════════════════════════
# MAIN — Run all 3 problems
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("   GENETIC ALGORITHMS FOR OPTIMIZATION")
    print("═"*55)

    results = {}

    # 1. Function Optimization
    print("\n[1/3] Function Optimization...")
    t = time.time()
    r1 = FunctionOptimizationGA(pop_size=100, generations=200).run()
    print(f"  ✅ Best x      = {r1['best_x']}")
    print(f"  ✅ Best f(x)   = {r1['best_fitness']}")
    print(f"  ⏱  Time        = {time.time()-t:.2f}s")
    results['function_optimization'] = r1

    # 2. Knapsack
    print("\n[2/3] Knapsack Problem...")
    t = time.time()
    r2 = KnapsackGA(pop_size=50, generations=150).run()
    print(f"  ✅ Packed items = {', '.join(r2['packed_items'])}")
    print(f"  ✅ Total Value  = ₹{r2['total_value']}")
    print(f"  ✅ Weight used  = {r2['total_weight']} / {r2['capacity']} kg")
    print(f"  ⏱  Time         = {time.time()-t:.2f}s")
    results['knapsack'] = r2

    # 3. TSP
    print("\n[3/3] Travelling Salesman (10 Indian Cities)...")
    t = time.time()
    r3 = TSPGA(pop_size=100, generations=500).run()
    print(f"  ✅ Route        = {' → '.join(r3['best_route'])}")
    print(f"  ✅ Distance     = {r3['best_distance_km']} km")
    print(f"  ⏱  Time         = {time.time()-t:.2f}s")
    results['tsp'] = r3

    print("\n" + "═"*55)
    print("   ALL 3 PROBLEMS SOLVED ✅")
    print("═"*55 + "\n")

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("📁 results.json saved\n")

    return results


if __name__ == '__main__':
    main()
