import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle
import json
import os
import argparse

def load_graph(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return nx.DiGraph()

def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

def get_existing_files(G):
    return set(data['file'] for _, data in G.nodes(data=True) if 'file' in data)

def load_new_chunks_and_embeddings(corpora_dir, existing_files):
    all_chunks = []
    all_embeddings = []

    for root, dirs, files in os.walk(corpora_dir):
        for filename in files:
            if filename.endswith('_chunks.json'):
                base_name = filename[:-12]  # Remove '_chunks.json'
                if base_name in existing_files:
                    continue  # Skip files already in the graph
                chunks_file = os.path.join(root, filename)
                embeddings_file = os.path.join(root, f"{base_name}_embeddings.npy")

                if os.path.exists(embeddings_file):
                    with open(chunks_file, 'r') as f:
                        chunks = json.load(f)
                    embeddings = np.load(embeddings_file)

                    # find the metadata for this node
                    metadata_file = os.path.join(root, f"{base_name}.txt.met")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    for i, chunk in enumerate(chunks):
                        chunk['file'] = base_name  # Add filename to chunk data
                        # iterate through the metadata and add it to the chunk
                        for key, value in metadata.items():
                            chunk[key] = value
                        all_chunks.append(chunk)
                        all_embeddings.append(embeddings[i])
                        # print(f" Chunk : {chunk}")
                        # print(f" Metadata : {metadata}")
                        # print(f" embeddings : {embeddings[i]}")
                else:
                    print(f"Embeddings file {embeddings_file} not found. Skipping.")
    return all_chunks, np.array(all_embeddings)

def add_new_nodes_and_edges(G, new_chunks, new_embeddings, n_neighbors=5):
    existing_node_count = len(G.nodes())
    print(f"Existing node count: {existing_node_count}")
    
    # Add new nodes
    for i, chunk in enumerate(tqdm(new_chunks, desc="Adding new nodes", unit="node")):
        new_node_id = existing_node_count + i
        G.add_node(new_node_id)
        for key, value in chunk.items():
            G.nodes[new_node_id][key] = value
        
        # add embedding to the node
        G.nodes[new_node_id]['embedding'] = new_embeddings[i]

        print(f"Sample new node data (node {new_node_id}):")
        for key, value in G.nodes[new_node_id].items():
            print(f"{key}: {value}")


    # Check if existing nodes have embeddings
    existing_embeddings = []
    for n in range(existing_node_count):
        if 'embedding' in G.nodes[n]:
            existing_embeddings.append(G.nodes[n]['embedding'])
        else:
            print(f"Warning: Node {n} doesn't have an embedding. Skipping.")

    if existing_embeddings:
        all_embeddings = np.vstack([np.array(existing_embeddings), new_embeddings])
    else:
        print("No existing embeddings found. Using only new embeddings.")
        all_embeddings = new_embeddings

    # Compute nearest neighbors for new nodes
    print("Computing nearest neighbors for new nodes...")
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors+1, len(all_embeddings)), algorithm='brute', metric='cosine').fit(all_embeddings)
    new_node_embeddings = all_embeddings[existing_node_count:]
    distances, indices = nbrs.kneighbors(new_node_embeddings)

    # Add edges for new nodes
    print("Adding edges for new nodes...")
    for i, (distances_i, indices_i) in enumerate(tqdm(zip(distances, indices), total=len(new_chunks))):
        new_node_id = existing_node_count + i
        for j, dist in zip(indices_i[1:], distances_i[1:]):  # Skip the first one (itself)
            if j < len(G.nodes()):  # Ensure we're not trying to add edges to non-existent nodes
                similarity = 1 - dist
                G.add_edge(new_node_id, j, weight=similarity)
                G.add_edge(j, new_node_id, weight=similarity)  # Add reverse edge for symmetry

    # count the number of nodes in the graph
    new_node_count = len(G.nodes()) - existing_node_count
    print(f"Added {new_node_count} new nodes")
    return G

def main(corpora_dir, output_file, n_neighbors):
    # Load existing graph or create a new one
    G = load_graph(output_file)
    existing_files = get_existing_files(G)

    # Load new chunks and embeddings
    new_chunks, new_embeddings = load_new_chunks_and_embeddings(corpora_dir, existing_files)

    if not new_chunks:
        print("No new files to add. Exiting.")
        return

    # Add new nodes and edges to the graph
    G = add_new_nodes_and_edges(G, new_chunks, new_embeddings, n_neighbors)

    # Save the updated graph
    save_graph(G, output_file)
    print(f"Graph updated and saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or update a graph from multiple chunk and embedding files")
    parser.add_argument("--corpora-dir", required=True, help="Directory containing corpora subfolders")
    parser.add_argument("--output", default="corpora_graph.gpickle", help="Output filename for the combined graph")
    parser.add_argument("--n-neighbors", type=int, default=5, help="Number of top similar neighbors to connect")
    args = parser.parse_args()

    main(args.corpora_dir, args.output, args.n_neighbors)