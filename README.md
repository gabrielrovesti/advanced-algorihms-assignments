# Advanced Algorithms Assignments

This repository contains implementations of classic graph algorithms across three assignments. Each implementation focuses on solving fundamental problems in graph theory with various approaches, analyzing their performance characteristics, and comparing different algorithmic strategies.

## Overview

The repository is organized into three main assignments:

1. **Minimum Spanning Tree Algorithms**
2. **Traveling Salesman Problem Algorithms**
3. **Minimum Cut Algorithms**

Each assignment includes a detailed implementation in Python, along with datasets for testing and a PDF describing the assignment requirements.

## Assignments

### Assignment 1: Minimum Spanning Tree Algorithms

Implementation of three algorithms for finding the Minimum Spanning Tree of an undirected weighted graph:

- **Prim's Algorithm** with heap optimization
- **Naïve Kruskal's Algorithm** with O(mn) complexity
- **Efficient Kruskal's Algorithm** using Union-Find data structure

The implementation compares the performance of these algorithms across different graph sizes and analyzes their behavior in relation to their theoretical complexity.

### Assignment 2: Traveling Salesman Problem Algorithms

Implementation of three approximation algorithms for the NP-hard Traveling Salesman Problem:

- **Nearest Neighbor** constructive heuristic
- **Farthest Insertion** constructive heuristic
- **MST-based 2-approximation algorithm**

The implementation analyzes solution quality (approximation ratio) and execution times across various problem instances.

### Assignment 3: Minimum Cut Algorithms

Implementation of three algorithms for finding the minimum cut in a weighted undirected graph:

- **Stoer-Wagner Algorithm** (deterministic)
- **Karger-Stein Algorithm** (randomized)
- **Hybrid Algorithm** (combining random contraction with deterministic finishing)

The implementation includes analysis of execution times, discovery times, and solution quality.

## Usage

Each assignment can be run independently. To execute an implementation:

1. Navigate to the corresponding assignment directory
2. Ensure you have the required dataset (extract the ZIP file if needed)
3. Run the Python implementation file:

```bash
# For Assignment 1
python mst_implementation.py

# For Assignment 2
python tsp_implementation.py

# For Assignment 3
python min_cut_algorithms.py
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Implementation Details

- All implementations use standard Python data structures (dictionaries, lists, sets)
- No external graph libraries are used (e.g., NetworkX); all graph algorithms are implemented from scratch
- Visualizations are generated using Matplotlib
- Performance measurements include execution time and solution quality metrics

## Results

Each implementation includes code to:
1. Run the algorithms on all test cases
2. Measure execution times
3. Analyze solution quality
4. Generate comparative plots
5. Output detailed results in CSV format

The performance characteristics observed in the experiments align with the theoretical complexity of the algorithms and provide practical insights into their behavior on various problem instances.