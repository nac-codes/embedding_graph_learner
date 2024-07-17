import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
import re
import argparse
import json
import os

def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def create_embedding_graph(all_chunks, all_embeddings, similarity_threshold=0.7):
    G = nx.DiGraph()
    edge_count = 0

    # Add nodes with a progress bar
    for i, chunk in enumerate(tqdm(all_chunks, desc="Adding nodes", unit="node")):
        G.add_node(i, content=chunk['content'], pages=chunk['pages'], file=chunk['file'], embedding=all_embeddings[i])

    total_comparisons = len(all_chunks) * (len(all_chunks) - 1) // 2
    with tqdm(total=total_comparisons, desc="Comparing chunks", unit="comparison") as pbar:
        for i in range(len(all_chunks)):
            for j in range(i+1, len(all_chunks)):
                cosine_similarity = 1 - cosine(all_embeddings[i], all_embeddings[j])

                if cosine_similarity > similarity_threshold:
                    G.add_edge(i, j, weight=cosine_similarity)
                    edge_count += 1
                
                pbar.update(1)

    print(f"Added {edge_count} edges to the graph")
    return G

def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_chunks_and_embeddings(directory):
    all_chunks = []
    all_embeddings = []

    for filename in os.listdir(directory):
        if filename.endswith('_chunks.json'):
            base_name = filename[:-12]  # Remove '_chunks.json'
            chunks_file = os.path.join(directory, filename)
            embeddings_file = os.path.join(directory, f"{base_name}_embeddings.npy")

            if os.path.exists(embeddings_file):
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                embeddings = np.load(embeddings_file)

                for chunk in chunks:
                    chunk['file'] = base_name  # Add filename to chunk data

                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)

    return all_chunks, np.array(all_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a graph from multiple chunk and embedding files")
    parser.add_argument("--directory", default=".", help="Directory containing chunk and embedding files")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Similarity threshold for creating edges in the graph")
    parser.add_argument("--output", default="combined_graph.gpickle", help="Output filename for the combined graph")
    args = parser.parse_args()

    print("Loading chunk data and embeddings...")
    all_chunks, all_embeddings = load_chunks_and_embeddings(args.directory)

    print(f"Loaded {len(all_chunks)} chunks from {len(set(chunk['file'] for chunk in all_chunks))} files")

    print("Creating the graph...")
    G = create_embedding_graph(all_chunks, all_embeddings, args.similarity_threshold)

    print("Saving the graph...")
    save_graph(G, args.output)

    print(f"Graph created and saved as {args.output}")

    # To load the graph later:
    # loaded_G = load_graph("combined_graph.gpickle")