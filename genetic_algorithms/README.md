# Genetic Algorithms for Optimization

A complete implementation of **Genetic Algorithms (GA)** from scratch using Python to solve three classic real-world optimization problems without using any machine learning libraries.

This project demonstrates how evolutionary algorithms inspired by natural selection can efficiently search for optimal solutions in complex problem spaces.

---

# Problems Solved

| # | Problem | Objective |
|---|----------|------------|
| 1 | Function Optimization | Find the value of `x` that maximizes a mathematical function |
| 2 | Knapsack Problem | Select items with maximum value under a weight limit |
| 3 | Travelling Salesman Problem (TSP) | Find the shortest route across multiple cities |

---

# Genetic Algorithm Concepts Used

- Population Initialization
- Fitness Function
- Tournament Selection
- Crossover
- Mutation
- Elitism
- Generational Evolution

---

# Problem 1 — Function Optimization

## Objective

Maximize the function:

```math
f(x) = sin(x) * cos(x/2) + x/10
```

## Features

- Real-valued chromosome representation
- Continuous optimization
- Random mutation
- Fitness tracking across generations

## Output

- Best value of `x`
- Maximum fitness value

---

# Problem 2 — Knapsack Problem

## Objective

Choose the best combination of items without exceeding the weight limit.

## Constraints

- Maximum capacity: **10 kg**
- Each item has:
  - Weight
  - Value

## Features

- Binary chromosome representation
- Penalizes overweight solutions
- Maximizes total value

## Output

- Selected items
- Total weight
- Total value

---

# Problem 3 — Travelling Salesman Problem (TSP)

## Objective

Find the shortest route that visits all cities exactly once and returns to the starting city.

## Dataset

The implementation uses **10 Indian cities** with predefined coordinates.

## Features

- Permutation-based chromosomes
- Ordered crossover
- Swap mutation
- Distance minimization fitness

## Output

- Best route
- Minimum travel distance

---

# Project Structure

```text
genetic_algorithms/
│
├── genetic_algorithm.py    # Main implementation
├── requirements.txt        # Required dependencies
├── README.md               # Documentation
└── results.json            # Auto-generated output
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/genetic_algorithms.git
cd genetic_algorithms
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
python genetic_algorithm.py
```

---

# Output

After execution:

- Results for all 3 problems are displayed in the terminal
- `results.json` is automatically generated

Example outputs include:

- Optimized function value
- Best knapsack item selection
- Shortest TSP route

---

# Adjustable Parameters

You can tune the Genetic Algorithm inside `genetic_algorithm.py`.

| Parameter | Default | Description |
|-----------|----------|-------------|
| `pop_size` | 100 | Number of candidate solutions |
| `generations` | 200 | Number of evolution cycles |
| `mutation_rate` | 0.01–0.02 | Mutation probability |
| `crossover_rate` | 0.8 | Parent crossover probability |
| `elite_size` | 2 | Best solutions preserved |

---

# Genetic Algorithm Workflow

```text
Initialize Population
        ↓
Evaluate Fitness
        ↓
Select Parents
        ↓
Perform Crossover
        ↓
Apply Mutation
        ↓
Create New Generation
        ↓
Repeat Until Termination
```

---

# Technologies Used

- Python 3
- NumPy
- Random
- Math
- JSON

---

# Applications of Genetic Algorithms

Genetic Algorithms are used in:

- Route Optimization
- Scheduling Systems
- Robotics
- Engineering Design
- Supply Chain Optimization
- Finance
- Artificial Intelligence

---

# Future Improvements

Possible enhancements:

- Fitness evolution graphs
- Parallel genetic algorithms
- Adaptive mutation
- GUI visualization
- Multi-objective optimization

---

# Author

Developed as a practical project to understand and implement Genetic Algorithms using Python.

---

# License

This project is open-source and available under the MIT License.
