import networkx as nx
import random
from tqdm import tqdm

def random_walk(G):
    visited = set()
    path = []
    stack = []

    def dfs(node, depth=0):
        visited.add(node)
        path.append(node)
        print(f"{'  ' * depth}Visiting node {node}")

        # Get unvisited neighbors
        neighbors = list(G.neighbors(node))
        unvisited = [n for n in neighbors if n not in visited]

        while unvisited:
            # Choose a random unvisited neighbor
            next_node = random.choice(unvisited)
            print(f"{'  ' * depth}Moving from node {node} to {next_node}")
            stack.append(node)
            dfs(next_node, depth + 1)
            unvisited = [n for n in neighbors if n not in visited]

        if stack:
            # Backtrack
            parent = stack.pop()
            print(f"{'  ' * depth}Backtracking from node {node} to {parent}")
            path.append(parent)

    # Start with a random node
    all_nodes = list(G.nodes())
    unvisited_nodes = [node for node in all_nodes if node not in visited]
    start_node = random.choice(unvisited_nodes)
    print(f"Starting walk from node {start_node}")
    dfs(start_node)

    # If there are disconnected components, visit them too
    while len(visited) < len(G.nodes()):
        unvisited_nodes = set(G.nodes()) - visited
        next_start = random.choice(list(unvisited_nodes))
        print(f"\nStarting new walk from disconnected node {next_start}")
        dfs(next_start)

    return path, len(visited)

def walk_and_print(G):
    path, nodes_visited = random_walk(G)
    print(f"\nRandom walk complete.")
    print(f"Total nodes visited: {nodes_visited}")
    print(f"Total nodes in graph: {len(G.nodes())}")
    # print total edges in graph
    print(f"Total edges in graph: {len(G.edges())}")
    return nodes_visited

if __name__ == "__main__":
    import pickle

    print("Loading the graph...")
    with open("embedding_graph.gpickle", 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    print("\nStarting random walk...")
    nodes_visited = walk_and_print(G)

    if nodes_visited == len(G.nodes()):
        print("\nSuccessfully visited all nodes in the graph!")
    else:
        print(f"\nWarning: Only visited {nodes_visited} out of {len(G.nodes())} nodes.")

    print("\nDone!")

    # walk for embedding_graph_clustered.gpickle
    print("Loading the clustered graph...")
    with open("embedding_graph_clustered.gpickle", 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded clustered graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    print("\nStarting random walk...")
    nodes_visited = walk_and_print(G)
    
    if nodes_visited == len(G.nodes()):
        print("\nSuccessfully visited all nodes in the clustered graph!")
    else:
        print(f"\nWarning: Only visited {nodes_visited} out of {len(G.nodes())} nodes.")
    
    print("\nDone!")