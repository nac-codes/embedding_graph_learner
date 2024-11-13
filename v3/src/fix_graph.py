import networkx as nx
import numpy as np
import json
import os
import pickle
import argparse
from tqdm import tqdm

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

def find_file_for_node(node_data, corpora_dir):
    file_name = node_data.get('file')
    if file_name:
        for root, _, files in os.walk(corpora_dir):
            if f"{file_name}_chunks.json" in files:
                return os.path.join(root, f"{file_name}_chunks.json"), os.path.join(root, f"{file_name}_embeddings.npy"), os.path.join(root, f"{file_name}.txt.met")
    return None, None, None

def load_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def fix_node(node, node_data, chunks_file, embeddings_file, metadata_file, chunk_index):
    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    
    # Load metadata
    metadata = load_metadata(metadata_file)
    
    # Update node data
    if chunk_index < len(chunks) and chunk_index < len(embeddings):
        chunk = chunks[chunk_index]
        for key, value in chunk.items():
            node_data[key] = value
        for key, value in metadata.items():
            node_data[key] = value
        node_data['embedding'] = embeddings[chunk_index].tolist()
    else:
        print(f"Warning: Chunk index {chunk_index} out of range for node {node}")

def fix_graph(G, corpora_dir):
    nodes_to_fix = []
    for node, data in G.nodes(data=True):
        if 'embedding' not in data or 'author' not in data:
            nodes_to_fix.append(node)
    
    print(f"Found {len(nodes_to_fix)} nodes to fix")
    
    for node in tqdm(nodes_to_fix, desc="Fixing nodes"):
        chunks_file, embeddings_file, metadata_file = find_file_for_node(G.nodes[node], corpora_dir)
        if chunks_file and embeddings_file:
            fix_node(node, G.nodes[node], chunks_file, embeddings_file, metadata_file, G.nodes[node].get('chunk_index', 0))
        else:
            print(f"Warning: Could not find files for node {node}")
    
    return G

def main(graph_file, corpora_dir, output_file):
    print(f"Loading graph from {graph_file}")
    G = load_graph(graph_file)
    
    print(f"Fixing graph")
    G = fix_graph(G, corpora_dir)
    
    print(f"Saving fixed graph to {output_file}")
    save_graph(G, output_file)
    
    print("Graph fixing complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix a graph by adding missing metadata and embeddings")
    parser.add_argument("--graph_file", help="Input graph file (.gpickle)")
    parser.add_argument("--corpora_dir", help="Directory containing corpora subfolders")
    parser.add_argument("--output", default="fixed_graph.gpickle", help="Output filename for the fixed graph")
    args = parser.parse_args()

    main(args.graph_file, args.corpora_dir, args.output)