import networkx as nx
import random
import pickle
import argparse
import os

def load_graph(graph_file_path):
    print(f"Loading the graph from {graph_file_path}...")
    with open(graph_file_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G

def sample_nodes(G, sample_size=5):
    return random.sample(list(G.nodes()), min(sample_size, len(G.nodes())))

def display_node_content(G, node):
    data = G.nodes[node]
    print(f"\n--- Node {node} ---")
    print(f"Title: {data.get('title', 'Unknown')}")
    print(f"Author: {data.get('author', 'Unknown')}")
    print(f"Date: {data.get('publication_date', 'Unknown')}")
    print(f"Pages: {data.get('pages', 'Unknown')}")
    print("\nContent:")
    print(data.get('content', 'No content available'))
    print("\n" + "-"*40)

def main():
    parser = argparse.ArgumentParser(description="Sample and display contents from a graph")
    parser.add_argument("graph_file", help="Path to the .gpickle graph file")
    parser.add_argument("-n", "--num_samples", type=int, default=5, help="Number of nodes to sample (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.graph_file):
        print(f"Graph file {args.graph_file} does not exist.")
        return

    G = load_graph(args.graph_file)
    sampled_nodes = sample_nodes(G, args.num_samples)

    print(f"\nDisplaying content for {len(sampled_nodes)} randomly sampled nodes:")
    for node in sampled_nodes:
        display_node_content(G, node)

if __name__ == "__main__":
    main()