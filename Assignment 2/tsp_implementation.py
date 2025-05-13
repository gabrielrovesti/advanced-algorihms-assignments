import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq
import os

# Distance calculation functions
def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance and round to nearest integer."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(dist)

def geo_distance(x1, y1, x2, y2):
    """Calculate geographic distance according to TSPLIB."""
    # Convert to radians, using only integer part as specified in the FAQ
    lat1 = deg2rad(int(x1))
    lon1 = deg2rad(int(y1))
    lat2 = deg2rad(int(x2))
    lon2 = deg2rad(int(y2))
    
    # Calculate distance according to the FAQ
    RRR = 6378.388  # Earth radius
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

class TSP:
    def __init__(self, filename):
        self.filename = filename
        self.name = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.optimal_solution = None
        self.coords = {}
        self.distances = {}
        self.parse_file()
        self.compute_distances()
    
    def parse_file(self):
        """Parse the TSP file."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("NAME"):
                    self.name = line.split(":")[1].strip()
                elif line.startswith("DIMENSION"):
                    self.dimension = int(line.split(":")[1].strip())
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    self.edge_weight_type = line.split(":")[1].strip()
                elif line.startswith("NODE_COORD_SECTION"):
                    i += 1
                    while i < len(lines) and not lines[i].startswith("EOF"):
                        node_data = lines[i].strip().split()
                        if len(node_data) >= 3:
                            node_id = int(node_data[0])
                            x = float(node_data[1])
                            y = float(node_data[2])
                            self.coords[node_id] = (x, y)
                        i += 1
                i += 1
    
    def compute_distances(self):
        """Compute distances between all pairs of nodes."""
        for i in self.coords:
            for j in self.coords:
                if i != j:
                    x1, y1 = self.coords[i]
                    x2, y2 = self.coords[j]
                    if self.edge_weight_type == "EUC_2D":
                        self.distances[(i, j)] = euclidean_distance(x1, y1, x2, y2)
                    elif self.edge_weight_type == "GEO":
                        self.distances[(i, j)] = geo_distance(x1, y1, x2, y2)
    
    def get_distance(self, i, j):
        """Get distance between nodes i and j."""
        if (i, j) in self.distances:
            return self.distances[(i, j)]
        elif (j, i) in self.distances:
            return self.distances[(j, i)]
        else:
            # This should not happen for a complete graph
            return float('inf')
    
    def get_nodes(self):
        """Get all nodes."""
        return list(self.coords.keys())
    
    def get_tour_length(self, tour):
        """Calculate the total length of a tour."""
        total = 0
        for i in range(len(tour) - 1):
            total += self.get_distance(tour[i], tour[i + 1])
        # Add distance from last to first node to complete the cycle
        total += self.get_distance(tour[-1], tour[0])
        return total

# Algorithm implementations
def nearest_neighbor(tsp, start_node=None):
    """Nearest Neighbor heuristic for TSP."""
    nodes = tsp.get_nodes()
    if start_node is None:
        start_node = nodes[0]
    
    unvisited = set(nodes)
    unvisited.remove(start_node)
    
    tour = [start_node]
    current = start_node
    
    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda node: tsp.get_distance(current, node))
        
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

def farthest_insertion(tsp):
    """Farthest Insertion heuristic for TSP."""
    nodes = tsp.get_nodes()
    start_node = nodes[0]
    
    # Initialize subtour with first node
    subtour = [start_node]
    unvisited = set(nodes[1:])
    
    # Add second node that is farthest from the first
    farthest = max(unvisited, key=lambda node: tsp.get_distance(start_node, node))
    subtour.append(farthest)
    subtour.append(start_node)  # Complete the cycle
    unvisited.remove(farthest)
    
    while unvisited:
        # For each unvisited node, find its distance to the nearest node in the subtour
        farthest_node = None
        max_distance = -1
        best_insertion_index = -1
        
        for node in unvisited:
            min_distance = float('inf')
            for subtour_node in subtour[:-1]:  # Exclude the duplicate start node
                distance = tsp.get_distance(node, subtour_node)
                if distance < min_distance:
                    min_distance = distance
            
            if min_distance > max_distance:
                max_distance = min_distance
                farthest_node = node
        
        # Find the best place to insert the farthest node
        min_insertion_cost = float('inf')
        for i in range(len(subtour) - 1):
            # Calculate cost of inserting between subtour[i] and subtour[i+1]
            insertion_cost = (tsp.get_distance(subtour[i], farthest_node) + 
                              tsp.get_distance(farthest_node, subtour[i+1]) - 
                              tsp.get_distance(subtour[i], subtour[i+1]))
            
            if insertion_cost < min_insertion_cost:
                min_insertion_cost = insertion_cost
                best_insertion_index = i + 1
        
        # Insert farthest node at the best position
        subtour.insert(best_insertion_index, farthest_node)
        unvisited.remove(farthest_node)
    
    # Remove the duplicate start node at the end
    return subtour[:-1]

def mst_approx(tsp):
    """2-approximate algorithm based on MST (Christofides without matching)."""
    nodes = tsp.get_nodes()
    n = len(nodes)
    
    # Step 1: Construct MST using Prim's algorithm
    start_node = nodes[0]
    mst = []
    
    visited = {start_node}
    edges = [(tsp.get_distance(start_node, v), start_node, v) for v in nodes if v != start_node]
    heapq.heapify(edges)
    
    while edges and len(visited) < n:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v))
            for w in nodes:
                if w not in visited and w != v:
                    heapq.heappush(edges, (tsp.get_distance(v, w), v, w))
    
    # Step 2: Create an adjacency list from MST
    adj_list = defaultdict(list)
    for u, v in mst:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Step 3: Perform preorder traversal to get an approximate tour
    tour = []
    visited = set()
    
    def dfs(node):
        if node not in visited:
            tour.append(node)
            visited.add(node)
            for neighbor in adj_list[node]:
                dfs(neighbor)
    
    dfs(start_node)
    
    return tour

def run_algorithm(tsp, algorithm, algorithm_name):
    """Run an algorithm and measure execution time."""
    start_time = time.time()
    
    if algorithm_name == "Nearest Neighbor":
        tour = algorithm(tsp)
    elif algorithm_name == "Farthest Insertion":
        tour = algorithm(tsp)
    elif algorithm_name == "MST 2-Approximate":
        tour = algorithm(tsp)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    tour_length = tsp.get_tour_length(tour)
    
    return tour, tour_length, execution_time

def get_optimal_solution(filename):
    """Get the optimal solution length from the assignment description."""
    optimal_solutions = {
        "burma14.tsp": 3323,
        "ulysses16.tsp": 6859,
        "ulysses22.tsp": 7013,
        "eil51.tsp": 426,
        "berlin52.tsp": 7542,
        "kroD100.tsp": 21294,
        "kroA100.tsp": 21282,
        "ch150.tsp": 6528,
        "gr202.tsp": 40160,
        "gr229.tsp": 134602,
        "pcb442.tsp": 50778,
        "d493.tsp": 35002,
        "dsj1000.tsp": 18659688
    }
    
    base_name = os.path.basename(filename)
    return optimal_solutions.get(base_name, None)

def main():
    """Main function to run all algorithms on all datasets."""
    # List of dataset files
    datasets = [
        "burma14.tsp", "ulysses16.tsp", "ulysses22.tsp", "eil51.tsp", 
        "berlin52.tsp", "kroD100.tsp", "kroA100.tsp", "ch150.tsp", 
        "gr202.tsp", "gr229.tsp", "pcb442.tsp", "d493.tsp", "dsj1000.tsp"
    ]
    
    # Algorithms to run
    algorithms = [
        (nearest_neighbor, "Nearest Neighbor"),
        (farthest_insertion, "Farthest Insertion"),
        (mst_approx, "MST 2-Approximate")
    ]
    
    # Results table
    results = []
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        tsp = TSP(dataset)
        optimal = get_optimal_solution(dataset)
        
        dataset_results = {"Dataset": dataset, "Dimension": tsp.dimension, "Optimal": optimal}
        
        for algorithm, algorithm_name in algorithms:
            print(f"  Running {algorithm_name}...")
            tour, tour_length, execution_time = run_algorithm(tsp, algorithm, algorithm_name)
            
            relative_error = (tour_length - optimal) / optimal if optimal else None
            
            dataset_results[f"{algorithm_name} Length"] = tour_length
            dataset_results[f"{algorithm_name} Time"] = execution_time
            dataset_results[f"{algorithm_name} Error"] = relative_error
        
        results.append(dataset_results)
    
    # Print results table
    print("\nResults:")
    print("=" * 120)
    print("{:<15} {:<8} {:<10} {:<15} {:<10} {:<10} {:<15} {:<10} {:<10} {:<15} {:<10} {:<10}".format(
        "Dataset", "Size", "Optimal", 
        "NN Length", "NN Time", "NN Error",
        "FI Length", "FI Time", "FI Error",
        "MST Length", "MST Time", "MST Error"
    ))
    print("-" * 120)
    
    for result in results:
        print("{:<15} {:<8} {:<10} {:<15} {:<10.4f} {:<10.4f} {:<15} {:<10.4f} {:<10.4f} {:<15} {:<10.4f} {:<10.4f}".format(
            result["Dataset"], result["Dimension"], result["Optimal"],
            result["Nearest Neighbor Length"], result["Nearest Neighbor Time"], result["Nearest Neighbor Error"],
            result["Farthest Insertion Length"], result["Farthest Insertion Time"], result["Farthest Insertion Error"],
            result["MST 2-Approximate Length"], result["MST 2-Approximate Time"], result["MST 2-Approximate Error"]
        ))
    
    # Save results to CSV
    with open("tsp_results.csv", "w") as f:
        f.write("Dataset,Size,Optimal,NN Length,NN Time,NN Error,FI Length,FI Time,FI Error,MST Length,MST Time,MST Error\n")
        for result in results:
            f.write(f"{result['Dataset']},{result['Dimension']},{result['Optimal']}," +
                   f"{result['Nearest Neighbor Length']},{result['Nearest Neighbor Time']:.6f},{result['Nearest Neighbor Error']:.6f}," +
                   f"{result['Farthest Insertion Length']},{result['Farthest Insertion Time']:.6f},{result['Farthest Insertion Error']:.6f}," +
                   f"{result['MST 2-Approximate Length']},{result['MST 2-Approximate Time']:.6f},{result['MST 2-Approximate Error']:.6f}\n")
    
    # Plot performance comparison
    plot_results(results)

def plot_results(results):
    """Create plots to visualize algorithm performance."""
    dimensions = [result["Dimension"] for result in results]
    nn_errors = [result["Nearest Neighbor Error"] for result in results]
    fi_errors = [result["Farthest Insertion Error"] for result in results]
    mst_errors = [result["MST 2-Approximate Error"] for result in results]
    
    nn_times = [result["Nearest Neighbor Time"] for result in results]
    fi_times = [result["Farthest Insertion Time"] for result in results]
    mst_times = [result["MST 2-Approximate Time"] for result in results]
    
    # Sort by dimension
    sorted_indices = np.argsort(dimensions)
    dimensions = [dimensions[i] for i in sorted_indices]
    nn_errors = [nn_errors[i] for i in sorted_indices]
    fi_errors = [fi_errors[i] for i in sorted_indices]
    mst_errors = [mst_errors[i] for i in sorted_indices]
    nn_times = [nn_times[i] for i in sorted_indices]
    fi_times = [fi_times[i] for i in sorted_indices]
    mst_times = [mst_times[i] for i in sorted_indices]
    
    # Plot relative errors
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, nn_errors, 'o-', label="Nearest Neighbor")
    plt.plot(dimensions, fi_errors, 's-', label="Farthest Insertion")
    plt.plot(dimensions, mst_errors, '^-', label="MST 2-Approximate")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Relative Error')
    plt.title('TSP Algorithm Relative Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('tsp_errors.png')
    
    # Plot execution times
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, nn_times, 'o-', label="Nearest Neighbor")
    plt.plot(dimensions, fi_times, 's-', label="Farthest Insertion")
    plt.plot(dimensions, mst_times, '^-', label="MST 2-Approximate")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('TSP Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('tsp_times.png')
    
    # Log-log plot for execution times
    plt.figure(figsize=(10, 6))
    plt.loglog(dimensions, nn_times, 'o-', label="Nearest Neighbor")
    plt.loglog(dimensions, fi_times, 's-', label="Farthest Insertion")
    plt.loglog(dimensions, mst_times, '^-', label="MST 2-Approximate")
    plt.xlabel('Number of Nodes (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('TSP Algorithm Performance Comparison (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tsp_times_loglog.png')

if __name__ == "__main__":
    main()
