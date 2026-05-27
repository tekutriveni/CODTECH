# Genetic Algorithms for Optimization

3 real-world problems solved using GA from scratch (no ML library needed).

## Problems Covered

| # | Problem | Goal |
|---|---------|------|
| 1 | **Function Optimization** | Find x that maximizes f(x) = sin(x)·cos(x/2) + x/10 |
| 2 | **Knapsack Problem** | Best items to pack within 10 kg weight limit |
| 3 | **Travelling Salesman (TSP)** | Shortest route across 10 Indian cities |

## GA Concepts Used

- **Population** — group of random solutions
- **Fitness Function** — how good is each solution
- **Selection** — tournament selection (best of 3)
- **Crossover** — combine two parents → child
- **Mutation** — small random change to avoid local optima
- **Generations** — repeat until best solution found

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run
```bash
python genetic_algorithm.py
```

### Output
Terminal lo 3 problems ke results print avutayi + `results.json` save avutundi.

## File Structure

```
genetic_algorithms/
├── genetic_algorithm.py   ← Main file (all 3 GAs)
├── requirements.txt       ← Dependencies
├── README.md              ← This file
└── results.json           ← Auto-generated after run
```

## Tuning Parameters

`genetic_algorithm.py` lo ee values change chesthe different results vastai:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `pop_size` | 100 | Larger = better search, slower |
| `generations` | 200 | More = better result |
| `mutation_rate` | 0.01–0.02 | Higher = more exploration |
| `crossover_rate` | 0.8 | Probability of combining parents |
