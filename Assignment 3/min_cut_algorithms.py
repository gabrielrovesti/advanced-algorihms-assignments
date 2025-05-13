import random
import math
import time
import copy
import heapq
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Graph class for efficient representation of weighted undirected graphs
class Graph:
    def __init__(self, num_vertices=0):
        self.adj = defaultdict(dict)  # Adjacency list with weights
    
    def add_edge(self, u, v, w):
        """Add an edge between vertices u and v with weight w"""
        self.adj[u][v] = w
        self.adj[v][u] = w  # Undirected graph
    
    def get_weight(self, u, v):
        """Get the weight of the edge between u and v"""
        return self.adj[u].get(v, 0)
    
    def contract_edge(self, u, v):
        """Contract the edge (u, v) by merging v into u"""
        for w, weight in list(self.adj[v].items()):
            if w != u:  # Skip self-loops
                self.adj[u][w] = self.adj[u].get(w, 0) + weight
                self.adj[w][u] = self.adj[w].get(u, 0) + weight
                del self.adj[w][v]
        
        # Remove v from the graph
        del self.adj[v]
    
    def get_vertices(self):
        """Get all vertices in the graph"""
        return list(self.adj.keys())
    
    def get_vertex_count(self):
        """Get the number of vertices in the graph"""
        return len(self.adj)
    
    def get_cut_value(self, cut_set):
        """Calculate the value of a cut defined by cut_set"""
        value = 0
        for u in cut_set:
            for v, w in self.adj[u].items():
                if v not in cut_set:
                    value += w
        return value
    
    def copy(self):
        """Create a deep copy of the graph"""
        g_copy = Graph()
        g_copy.adj = copy.deepcopy(self.adj)
        return g_copy


# Stoer-Wagner Algorithm
def stoer_wagner(graph):
    """Stoer-Wagner algorithm for minimum cut
    
    Returns:
        - Minimum cut value
        - One side of the minimum cut
    """
    vertices = graph.get_vertices()
    n = len(vertices)
    
    if n <= 1:
        return 0, []
    
    # Create a copy of the graph
    g = graph.copy()
    
    # For tracking the best cut
    best_cut_value = float('inf')
    best_cut = []
    
    # Map to track merged vertices (to identify the cut)
    merged = {v: [v] for v in vertices}
    
    # Run the algorithm n-1 times
    for _ in range(n-1):
        # Find a minimum s-t cut
        cut_value, s, t = minimum_cut_phase(g)
        
        # Update the best cut if necessary
        if cut_value < best_cut_value:
            best_cut_value = cut_value
            best_cut = merged[t]
        
        # Merge t into s
        merged[s].extend(merged[t])
        del merged[t]
        
        # Contract edge (s, t)
        g.contract_edge(s, t)
        
        if g.get_vertex_count() == 1:
            break
    
    return best_cut_value, best_cut

def minimum_cut_phase(graph):
    """Single phase of the Stoer-Wagner algorithm
    
    Finds a cut using the maximum adjacency search algorithm
    
    Returns:
        - Cut value
        - Second-to-last vertex added (s)
        - Last vertex added (t)
    """
    vertices = graph.get_vertices()
    if not vertices:
        return float('inf'), None, None
    
    # Start with an arbitrary vertex
    a = vertices[0]
    a_vertices = [a]
    cut_of_the_phase = {a}
    
    # Track the weight of edges connecting each vertex to the cut
    weights = {v: graph.get_weight(a, v) for v in vertices if v != a}
    
    # Use a priority queue for efficient selection
    pq = [(-weights.get(v, 0), v) for v in vertices if v != a]
    heapq.heapify(pq)
    
    # Perform maximum adjacency search
    while len(a_vertices) < len(vertices):
        # Extract the most tightly connected vertex
        next_vertex = None
        while pq:
            neg_weight, v = heapq.heappop(pq)
            if v not in cut_of_the_phase:
                next_vertex = v
                break
        
        if next_vertex is None:
            break
        
        # Add the vertex to the cut
        cut_of_the_phase.add(next_vertex)
        a_vertices.append(next_vertex)
        
        # Update the weights and priority queue
        for v, w in graph.adj[next_vertex].items():
            if v not in cut_of_the_phase:
                weights[v] = weights.get(v, 0) + w
                heapq.heappush(pq, (-weights[v], v))
    
    # Last two vertices added are s and t
    s, t = a_vertices[-2], a_vertices[-1]
    
    # Calculate the cut value
    cut_value = sum(graph.get_weight(t, v) for v in graph.adj[t])
    
    return cut_value, s, t


# Karger-Stein Algorithm
def karger_stein(graph, repetitions=None):
    """Karger-Stein randomized algorithm for minimum cut
    
    Args:
        graph: The graph to find the minimum cut in
        repetitions: Number of repetitions to run the algorithm
    
    Returns:
        - Minimum cut value
        - One side of the minimum cut
        - Time at which the minimum cut was discovered
    """
    vertices = graph.get_vertices()
    n = len(vertices)
    
    if repetitions is None:
        # Calculate repetitions for probability at least 1-1/n
        repetitions = min(int(math.ceil(math.log(n) * n)), 100)
    
    best_cut_value = float('inf')
    best_cut = []
    discovery_time = None
    start_time = time.time()
    
    for _ in range(repetitions):
        g_copy = graph.copy()
        cut_value, cut = recursive_contract(g_copy)
        
        if cut_value < best_cut_value:
            best_cut_value = cut_value
            best_cut = cut
            discovery_time = time.time() - start_time
        
        # Check timeout (10 minutes)
        if time.time() - start_time > 600:
            break
    
    return best_cut_value, best_cut, discovery_time

def recursive_contract(graph):
    """Recursive function for Karger-Stein algorithm
    
    Args:
        graph: The graph to contract
    
    Returns:
        - Cut value
        - One side of the cut
    """
    n = graph.get_vertex_count()
    
    # Base case
    if n <= 2:
        vertices = graph.get_vertices()
        if n == 2:
            u, v = vertices
            return graph.get_weight(u, v), [u]
        else:
            return 0, vertices
    
    # Calculate t = ceil(n / sqrt(2)) + 1
    t = int(math.ceil(n / math.sqrt(2))) + 1
    
    # Contract the graph until t vertices remain
    g_contracted = contract_graph_to_size(graph, t)
    
    # Make two recursive calls
    cut_value1, cut1 = recursive_contract(g_contracted.copy())
    cut_value2, cut2 = recursive_contract(g_contracted.copy())
    
    # Return the better of the two cuts
    if cut_value1 <= cut_value2:
        return cut_value1, cut1
    else:
        return cut_value2, cut2

def contract_graph_to_size(graph, target_size):
    """Contract the graph until it has target_size vertices
    
    Args:
        graph: The graph to contract
        target_size: Target number of vertices
    
    Returns:
        Contracted graph
    """
    g = graph.copy()
    
    # Contract until target_size vertices remain
    while g.get_vertex_count() > target_size:
        # Choose a random edge to contract
        vertices = g.get_vertices()
        u = random.choice(vertices)
        
        if not g.adj[u]:  # Skip isolated vertices
            continue
        
        v = random.choice(list(g.adj[u].keys()))
        g.contract_edge(u, v)
    
    return g


# Hybrid Algorithm
def hybrid_algorithm(graph, repetitions=None):
    """Hybrid algorithm that combines Karger-Stein contraction with Stoer-Wagner
    
    Args:
        graph: The graph to find the minimum cut in
        repetitions: Number of repetitions to run the algorithm
    
    Returns:
        - Minimum cut value
        - One side of the minimum cut
        - Time at which the minimum cut was discovered
    """
    vertices = graph.get_vertices()
    n = len(vertices)
    
    if repetitions is None:
        # Calculate repetitions for probability at least 1-1/n
        repetitions = min(int(math.ceil(math.log(n) * n)), 100)
    
    best_cut_value = float('inf')
    best_cut = []
    discovery_time = None
    start_time = time.time()
    
    for _ in range(repetitions):
        g_copy = graph.copy()
        
        # Contract the graph to size t = ceil(n / sqrt(2)) + 1
        t = int(math.ceil(n / math.sqrt(2))) + 1
        g_contracted = contract_graph_to_size(g_copy, t)
        
        # Apply Stoer-Wagner on the contracted graph
        cut_value, cut = stoer_wagner(g_contracted)
        
        if cut_value < best_cut_value:
            best_cut_value = cut_value
            best_cut = cut
            discovery_time = time.time() - start_time
        
        # Check timeout (10 minutes)
        if time.time() - start_time > 600:
            break
    
    return best_cut_value, best_cut, discovery_time


# Utility functions
def read_graph_from_file(filename):
    """Read a graph from a file
    
    The file should have the format:
    [number_of_nodes] [number_of_edges]
    [node1] [node2] [weight]
    ...
    
    Returns:
        The graph
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # First line contains number of vertices and edges
        parts = lines[0].strip().split()
        num_vertices = int(parts[0])
        
        graph = Graph()
        
        # Read edges
        for i in range(1, len(lines)):
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                u = int(parts[0])
                v = int(parts[1])
                w = int(parts[2])
                graph.add_edge(u, v, w)
        
        return graph

def run_with_timeout(func, args=(), timeout=600):
    """Run a function with a timeout
    
    Args:
        func: The function to run
        args: The arguments to pass to the function
        timeout: Timeout in seconds
    
    Returns:
        - The result of the function
        - The execution time
    """
    start_time = time.time()
    result = None
    
    try:
        result = func(*args)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    execution_time = time.time() - start_time
    
    # Check if timeout occurred
    if execution_time > timeout:
        print(f"  Algorithm timed out after {timeout} seconds")
    
    return result, execution_time

def run_experiments(filenames, timeout=600):
    """Run experiments on the given filenames
    
    Args:
        filenames: List of filenames
        timeout: Timeout for each algorithm in seconds
    
    Returns:
        List of results for each file
    """
    results = []
    
    for filename in filenames:
        print(f"Processing {filename}...")
        graph = read_graph_from_file(filename)
        
        n = graph.get_vertex_count()
        
        # Calculate repetitions for Karger-Stein and Hybrid
        repetitions = min(int(math.ceil(math.log(n) * n)), 100)  # Cap repetitions
        
        # Run Stoer-Wagner
        print("  Running Stoer-Wagner...")
        sw_result, sw_time = run_with_timeout(stoer_wagner, args=(graph,), timeout=timeout)
        if sw_result:
            sw_cut_value, sw_cut = sw_result
        else:
            sw_cut_value, sw_cut = float('inf'), []
        
        # Run Karger-Stein
        print("  Running Karger-Stein...")
        ks_result, ks_time = run_with_timeout(karger_stein, args=(graph, repetitions), timeout=timeout)
        if ks_result:
            ks_cut_value, ks_cut, ks_discovery_time = ks_result
        else:
            ks_cut_value, ks_cut, ks_discovery_time = float('inf'), [], None
        
        # Run Hybrid
        print("  Running Hybrid...")
        hybrid_result, hybrid_time = run_with_timeout(hybrid_algorithm, args=(graph, repetitions), timeout=timeout)
        if hybrid_result:
            hybrid_cut_value, hybrid_cut, hybrid_discovery_time = hybrid_result
        else:
            hybrid_cut_value, hybrid_cut, hybrid_discovery_time = float('inf'), [], None
        
        results.append({
            'filename': filename,
            'vertices': n,
            'sw_cut': sw_cut_value,
            'sw_time': sw_time,
            'ks_cut': ks_cut_value,
            'ks_time': ks_time,
            'ks_discovery_time': ks_discovery_time,
            'hybrid_cut': hybrid_cut_value,
            'hybrid_time': hybrid_time,
            'hybrid_discovery_time': hybrid_discovery_time
        })
        
        # Print results for this graph
        print(f"  Stoer-Wagner: cut = {sw_cut_value}, time = {sw_time:.4f}s")
        print(f"  Karger-Stein: cut = {ks_cut_value}, time = {ks_time:.4f}s, discovery = {ks_discovery_time if ks_discovery_time else 'N/A'}")
        print(f"  Hybrid: cut = {hybrid_cut_value}, time = {hybrid_time:.4f}s, discovery = {hybrid_discovery_time if hybrid_discovery_time else 'N/A'}")
        
    return results

def visualize_results(results):
    """Visualize the results of the experiments
    
    Creates plots for:
    - Execution time vs number of vertices
    - Log-log plot of execution time vs number of vertices
    - Discovery time vs execution time for randomized algorithms
    
    Args:
        results: List of results from run_experiments
    """
    # Extract data for plotting
    vertices = [r['vertices'] for r in results]
    sw_times = [r['sw_time'] for r in results]
    ks_times = [r['ks_time'] for r in results]
    hybrid_times = [r['hybrid_time'] for r in results]
    
    # Sort by number of vertices
    sorted_indices = np.argsort(vertices)
    vertices = [vertices[i] for i in sorted_indices]
    sw_times = [sw_times[i] for i in sorted_indices]
    ks_times = [ks_times[i] for i in sorted_indices]
    hybrid_times = [hybrid_times[i] for i in sorted_indices]
    
    # Create execution time plot
    plt.figure(figsize=(12, 8))
    plt.plot(vertices, sw_times, 'o-', label='Stoer-Wagner')
    plt.plot(vertices, ks_times, 's-', label='Karger-Stein')
    plt.plot(vertices, hybrid_times, '^-', label='Hybrid')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_times.png')
    
    # Create log-log plot for better visualization
    plt.figure(figsize=(12, 8))
    plt.loglog(vertices, sw_times, 'o-', label='Stoer-Wagner')
    plt.loglog(vertices, ks_times, 's-', label='Karger-Stein')
    plt.loglog(vertices, hybrid_times, '^-', label='Hybrid')
    plt.xlabel('Number of Vertices (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('Algorithm Performance Comparison (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_times_loglog.png')
    
    # Extract discovery times for randomized algorithms
    valid_ks = [(r['ks_time'], r['ks_discovery_time']) for r in results if r['ks_discovery_time'] is not None]
    ks_times_with_discovery = [t for t, _ in valid_ks]
    ks_discovery = [d for _, d in valid_ks]
    
    valid_hybrid = [(r['hybrid_time'], r['hybrid_discovery_time']) for r in results if r['hybrid_discovery_time'] is not None]
    hybrid_times_with_discovery = [t for t, _ in valid_hybrid]
    hybrid_discovery = [d for _, d in valid_hybrid]
    
    # Create discovery time vs execution time plot
    plt.figure(figsize=(12, 8))
    
    if ks_discovery:
        plt.subplot(1, 2, 1)
        plt.scatter(ks_times_with_discovery, ks_discovery)
        max_time = max(ks_times_with_discovery)
        plt.plot([0, max_time], [0, max_time], 'r--')  # Diagonal line
        plt.xlabel('Total Execution Time (seconds)')
        plt.ylabel('Discovery Time (seconds)')
        plt.title('Karger-Stein Algorithm')
        plt.grid(True)
    
    if hybrid_discovery:
        plt.subplot(1, 2, 2)
        plt.scatter(hybrid_times_with_discovery, hybrid_discovery)
        max_time = max(hybrid_times_with_discovery)
        plt.plot([0, max_time], [0, max_time], 'r--')  # Diagonal line
        plt.xlabel('Total Execution Time (seconds)')
        plt.ylabel('Discovery Time (seconds)')
        plt.title('Hybrid Algorithm')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('discovery_times.png')
    
    # Create a table of min-cut values
    cut_values = []
    for r in results:
        filename = r['filename']
        sw_cut = r['sw_cut']
        ks_cut = r['ks_cut']
        hybrid_cut = r['hybrid_cut']
        
        cut_values.append([filename, r['vertices'], sw_cut, ks_cut, hybrid_cut])
    
    cut_values.sort(key=lambda x: x[1])  # Sort by number of vertices
    
    # Print table
    print("\nMin-Cut Values:")
    print("{:<30} {:<10} {:<15} {:<15} {:<15}".format(
        "Filename", "Vertices", "Stoer-Wagner", "Karger-Stein", "Hybrid"))
    print("-" * 85)
    
    for row in cut_values:
        print("{:<30} {:<10} {:<15} {:<15} {:<15}".format(
            row[0], row[1], row[2], row[3], row[4]))
    
    # Save results to CSV
    with open('min_cut_results.csv', 'w') as f:
        f.write("filename,vertices,stoer_wagner_cut,karger_stein_cut,hybrid_cut,sw_time,ks_time,hybrid_time,ks_discovery,hybrid_discovery\n")
        
        for r in results:
            f.write(f"{r['filename']},{r['vertices']},{r['sw_cut']},{r['ks_cut']},{r['hybrid_cut']},{r['sw_time']:.6f},{r['ks_time']:.6f},{r['hybrid_time']:.6f},{r['ks_discovery_time'] if r['ks_discovery_time'] else 'N/A'},{r['hybrid_discovery_time'] if r['hybrid_discovery_time'] else 'N/A'}\n")

def main():
    """Main function to run experiments on all files matching the pattern"""
    import glob
    import os
    
    # Assume input files are in the current directory
    filenames = sorted(glob.glob("input_random_*.txt"))
    
    if not filenames:
        print("No input files found. Please make sure input files are in the current directory.")
        return
    
    print(f"Found {len(filenames)} input files.")
    results = run_experiments(filenames)
    visualize_results(results)
    
    print("\nExperiments completed. Results saved to 'min_cut_results.csv'.")
    print("Plots saved as 'execution_times.png', 'execution_times_loglog.png', and 'discovery_times.png'.")

if __name__ == "__main__":
    main()
