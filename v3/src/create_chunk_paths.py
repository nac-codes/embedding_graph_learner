import os
import json
import pickle
import argparse
from tqdm import tqdm

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

def find_matching_node(G, chunk_content):
    for node, data in G.nodes(data=True):
        if data.get('content') == chunk_content:
            return node
    return None

def process_chunks(corpora_dir, graph_file):
    # Load the graph
    print(f"Loading graph from {graph_file}")
    G = load_graph(graph_file)    
    
    # Process each subfolder in corpora directory
    for subdir in tqdm(os.listdir(corpora_dir), desc="Processing folders"):
        subdir_path = os.path.join(corpora_dir, subdir)
        if not os.path.isdir(subdir_path) or subdir == 'chunks':
            continue
        
        # Create chunks directory if it doesn't exist
        chunks_dir = os.path.join(subdir_path, 'chunks')
        os.makedirs(chunks_dir, exist_ok=True)
            
        # Find _chunks.json file
        chunks_file = None
        for file in os.listdir(subdir_path):
            if file.endswith('_chunks.json'):
                chunks_file = os.path.join(subdir_path, file)
                break
                
        if not chunks_file:
            print(f"No chunks file found in {subdir_path}")
            continue
            
        # Load chunks
        print(f"Processing chunks from {chunks_file}")
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
            
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create chunk filename
            chunk_filename = f"{subdir}_{i:04d}.txt"
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            
            # Save chunk content to file
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
            
            # Find matching node in graph
            node = find_matching_node(G, chunk['content'])
            if node is not None:
                # Add chunk_file attribute to node
                G.nodes[node]['chunk_file'] = os.path.relpath(chunk_path, corpora_dir)
                print(f"Added chunk file path to node {node}: {G.nodes[node]['chunk_file']}")
            else:
                print(f"Warning: No matching node found for chunk {chunk_filename}")
    
    # Save updated graph
    print(f"Saving updated graph to {graph_file}")
    save_graph(G, graph_file)
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Extract chunks and update graph with chunk file paths")
    parser.add_argument("--corpora-dir", required=True, help="Directory containing corpora subfolders")
    parser.add_argument("--graph-file", default="corpora_graph.gpickle", help="Path to graph file")
    args = parser.parse_args()

    if not os.path.isdir(args.corpora_dir):
        print(f"The directory {args.corpora_dir} does not exist.")
        return

    if not os.path.exists(args.graph_file):
        print(f"The graph file {args.graph_file} does not exist.")
        return

    process_chunks(args.corpora_dir, args.graph_file)

if __name__ == "__main__":
    main()