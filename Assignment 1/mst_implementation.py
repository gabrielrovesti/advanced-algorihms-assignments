import time
import heapq
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices + 1)]  # +1 for 1-indexed vertices
    
    def add_edge(self, u, v, w):
        """Add an edge to the graph"""
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))
    
    def get_all_edges(self):
        """Return all edges of the graph without duplicates"""
        edges = []
        for u in range(1, self.V + 1):
            for v, w in self.graph[u]:
                if u < v:  # To avoid duplicates in undirected graph
                    edges.append((u, v, w))
        return edges

class DisjointSet:
    """Union-Find data structure with path compression and union by rank"""
    def __init__(self, n):
        self.parent = list(range(n + 1))  # +1 for 1-indexed vertices
        self.rank = [0] * (n + 1)
    
    def find(self, x):
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def prim_mst(graph):
    """
    Prim's algorithm implementation using min-heap
    Time Complexity: O(E log V)
    """
    total_weight = 0
    V = graph.V
    
    # Create a set to track vertices in MST
    in_mst = [False] * (V + 1)
    
    # Start from vertex 1
    start = 1
    in_mst[start] = True
    
    # Priority queue for edges
    pq = []
    
    # Add all edges from start vertex
    for v, w in graph.graph[start]:
        heapq.heappush(pq, (w, start, v))
    
    # Process edges until we have V-1 edges or no more edges
    mst_edges = 0
    
    while pq and mst_edges < V - 1:
        weight, u, v = heapq.heappop(pq)
        
        if in_mst[v]:
            continue
        
        # Add vertex v to MST
        in_mst[v] = True
        total_weight += weight
        mst_edges += 1
        
        # Add all edges from v
        for next_v, next_w in graph.graph[v]:
            if not in_mst[next_v]:
                heapq.heappush(pq, (next_w, v, next_v))
    
    # Check if we have a spanning tree
    if mst_edges != V - 1:
        print(f"Warning: Graph is not connected. MST includes only {mst_edges} edges.")
    
    return total_weight

def naive_kruskal_mst(graph):
    """
    Naive Kruskal's algorithm with O(EV) complexity
    """
    total_weight = 0
    V = graph.V
    
    # Get all edges and sort by weight
    edges = graph.get_all_edges()
    edges.sort(key=lambda x: x[2])
    
    # Store MST edges
    mst = []
    
    for u, v, w in edges:
        # Check if adding this edge would create a cycle
        if not creates_cycle(u, v, mst, V):
            mst.append((u, v, w))
            total_weight += w
            
            # Early termination
            if len(mst) == V - 1:
                break
    
    return total_weight

def creates_cycle(u, v, mst, V):
    """
    Check if adding edge (u,v) to MST would create a cycle
    This is the naive approach with O(V) complexity per edge
    """
    # Build adjacency list from current MST
    adj_list = defaultdict(list)
    for edge_u, edge_v, _ in mst:
        adj_list[edge_u].append(edge_v)
        adj_list[edge_v].append(edge_u)
    
    # Check if v is reachable from u
    visited = [False] * (V + 1)
    
    def dfs(node):
        visited[node] = True
        if node == v:
            return True
        
        for neighbor in adj_list[node]:
            if not visited[neighbor] and dfs(neighbor):
                return True
        
        return False
    
    return dfs(u)

def efficient_kruskal_mst(graph):
    """
    Efficient Kruskal's algorithm using Union-Find
    Time Complexity: O(E log E) or O(E log V)
    """
    total_weight = 0
    V = graph.V
    
    # Get all edges and sort by weight
    edges = graph.get_all_edges()
    edges.sort(key=lambda x: x[2])
    
    # Initialize Union-Find
    ds = DisjointSet(V)
    
    # Store MST edges count
    mst_edge_count = 0
    
    for u, v, w in edges:
        # Check if adding this edge would create a cycle
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            total_weight += w
            mst_edge_count += 1
            
            # Early termination when we have V-1 edges
            if mst_edge_count == V - 1:
                break
    
    # Check if we have a spanning tree
    if mst_edge_count != V - 1:
        print(f"Warning: Graph is not connected. MST includes only {mst_edge_count} edges.")
    
    return total_weight

def read_graph_from_file(filename):
    """
    Read graph from file using the specified format
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # First line contains number of vertices and edges
        parts = lines[0].strip().split()
        num_vertices = int(parts[0])
        
        graph = Graph(num_vertices)
        
        # Read edges
        for i in range(1, len(lines)):
            parts = lines[i].strip().split()
            if len(parts) >= 3:  # Ensure the line has enough parts
                u = int(parts[0])
                v = int(parts[1])
                w = int(parts[2])
                graph.add_edge(u, v, w)
        
        return graph

def measure_performance(graph, algorithm_func, algorithm_name):
    """
    Measure execution time and MST weight for a given algorithm
    """
    print(f"  Running {algorithm_name}...")
    start_time = time.time()
    weight = algorithm_func(graph)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return weight, execution_time

def run_experiments(file_list):
    """
    Run experiments on all graph files
    """
    results = []
    
    for filename in sorted(file_list):
        base_filename = os.path.basename(filename)
        print(f"Processing {base_filename}...")
        try:
            graph = read_graph_from_file(filename)
            
            # Get number of vertices
            num_vertices = graph.V
            
            # For very large graphs, skip naive Kruskal
            run_naive = num_vertices <= 5000
            
            # Measure performance for each algorithm
            prim_weight, prim_time = measure_performance(graph, prim_mst, "Prim's algorithm")
            
            naive_kruskal_weight, naive_kruskal_time = None, None
            if run_naive:
                naive_kruskal_weight, naive_kruskal_time = measure_performance(
                    graph, naive_kruskal_mst, "Naive Kruskal's algorithm"
                )
            
            efficient_kruskal_weight, efficient_kruskal_time = measure_performance(
                graph, efficient_kruskal_mst, "Efficient Kruskal's algorithm"
            )
            
            # Verify that algorithms produce the same MST weight
            weights_match = True
            weights = [w for w in [prim_weight, naive_kruskal_weight, efficient_kruskal_weight] if w is not None]
            if len(set(weights)) > 1:
                weights_match = False
                print(f"Warning: Different MST weights for {base_filename}!")
                print(f"  Prim: {prim_weight}")
                if naive_kruskal_weight is not None:
                    print(f"  Naive Kruskal: {naive_kruskal_weight}")
                print(f"  Efficient Kruskal: {efficient_kruskal_weight}")
            
            results.append({
                'filename': base_filename,
                'num_vertices': num_vertices,
                'prim_weight': prim_weight,
                'prim_time': prim_time,
                'naive_kruskal_weight': naive_kruskal_weight,
                'naive_kruskal_time': naive_kruskal_time,
                'efficient_kruskal_weight': efficient_kruskal_weight,
                'efficient_kruskal_time': efficient_kruskal_time,
                'weights_match': weights_match
            })
            
            print(f"  MST weight: {prim_weight}")
            print(f"  Prim time: {prim_time:.6f}s")
            if run_naive:
                print(f"  Naive Kruskal time: {naive_kruskal_time:.6f}s")
            print(f"  Efficient Kruskal time: {efficient_kruskal_time:.6f}s")
            print()
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return results

def plot_results(results):
    """
    Create plots from the experimental results
    """
    # Extract data
    vertices = [r['num_vertices'] for r in results]
    prim_times = [r['prim_time'] for r in results]
    efficient_kruskal_times = [r['efficient_kruskal_time'] for r in results]
    
    # Extract data for naive Kruskal (only for smaller graphs)
    small_vertices = []
    naive_kruskal_times = []
    for r in results:
        if r['naive_kruskal_time'] is not None:
            small_vertices.append(r['num_vertices'])
            naive_kruskal_times.append(r['naive_kruskal_time'])
    
    # Sort by number of vertices
    sorted_indices = np.argsort(vertices)
    vertices = [vertices[i] for i in sorted_indices]
    prim_times = [prim_times[i] for i in sorted_indices]
    efficient_kruskal_times = [efficient_kruskal_times[i] for i in sorted_indices]
    
    # Sort data for naive Kruskal
    if small_vertices:
        small_sorted_indices = np.argsort(small_vertices)
        small_vertices = [small_vertices[i] for i in small_sorted_indices]
        naive_kruskal_times = [naive_kruskal_times[i] for i in small_sorted_indices]
    
    # Linear scale plot
    plt.figure(figsize=(12, 7))
    plt.plot(vertices, prim_times, 'o-', label="Prim's Algorithm")
    plt.plot(vertices, efficient_kruskal_times, '^-', label="Efficient Kruskal's Algorithm")
    if naive_kruskal_times:
        plt.plot(small_vertices, naive_kruskal_times, 's-', label="Naive Kruskal's Algorithm")
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (seconds)')
    plt.title('MST Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('mst_performance_linear.png')
    
    # Log-log scale plot
    plt.figure(figsize=(12, 7))
    plt.loglog(vertices, prim_times, 'o-', label="Prim's Algorithm")
    plt.loglog(vertices, efficient_kruskal_times, '^-', label="Efficient Kruskal's Algorithm")
    if naive_kruskal_times:
        plt.loglog(small_vertices, naive_kruskal_times, 's-', label="Naive Kruskal's Algorithm")
    plt.xlabel('Number of Vertices (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('MST Algorithm Performance Comparison (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('mst_performance_loglog.png')
    
    # Print results table
    print("\nResults Table:")
    print("{:<30} {:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "File", "Vertices", "Prim Time (s)", "Naive Kruskal (s)", "Efficient Kruskal (s)", "MST Weight"))
    print("-" * 100)
    
    for i in sorted_indices:
        r = results[i]
        naive_time = f"{r['naive_kruskal_time']:.6f}" if r['naive_kruskal_time'] is not None else "N/A"
        print("{:<30} {:<10d} {:<15.6f} {:<15s} {:<15.6f} {:<15d}".format(
            r['filename'], 
            r['num_vertices'],
            r['prim_time'],
            naive_time,
            r['efficient_kruskal_time'],
            r['prim_weight']))
    
    # Save results to CSV
    with open('mst_results.csv', 'w') as f:
        f.write("filename,vertices,prim_time,naive_kruskal_time,efficient_kruskal_time,mst_weight\n")
        for i in sorted_indices:
            r = results[i]
            naive_time = f"{r['naive_kruskal_time']}" if r['naive_kruskal_time'] is not None else "N/A"
            f.write(f"{r['filename']},{r['num_vertices']},{r['prim_time']},{naive_time},{r['efficient_kruskal_time']},{r['prim_weight']}\n")

def main():
    # Directory containing the graph files
    directory = "."  # Change this to your data directory
    
    # Get list of graph files
    file_list = []
    for filename in os.listdir(directory):
        if filename.startswith("input_random_") and filename.endswith(".txt"):
            file_list.append(os.path.join(directory, filename))
    
    if not file_list:
        print("No graph files found! Make sure the files are in the correct directory.")
        return
    
    print(f"Found {len(file_list)} graph files.")
    
    # Run experiments
    results = run_experiments(file_list)
    
    if results:
        # Plot and print results
        plot_results(results)
        print("\nResults saved to mst_results.csv")
        print("Plots saved as mst_performance_linear.png and mst_performance_loglog.png")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
