import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
import re
import argparse


def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def create_embedding_graph(chunk_data, chunk_embeddings, similarity_threshold=0.7):
    G = nx.DiGraph()
    edge_count = 0

    # Add nodes with a progress bar
    for i, chunk in enumerate(tqdm(chunk_data, desc="Adding nodes", unit="node")):
        G.add_node(i, content=chunk['content'], embedding=chunk_embeddings[i])

    total_comparisons = len(chunk_data) * (len(chunk_data) - 1) // 2
    with tqdm(total=total_comparisons, desc="Comparing chunks", unit="comparison") as pbar:
        for i in range(len(chunk_data)):
            for j in range(i+1, len(chunk_data)):
                cosine_similarity = 1 - cosine(chunk_embeddings[i], chunk_embeddings[j])                                

                # print(f"Comparing chunks {i} and {j}:")
                # print(f"Cosine similarity: {cosine_similarity}")
                # print(f"N-gram score: {ngram_score}")
                # print(f"Combined score: {combined_score}\n")
                if cosine_similarity > similarity_threshold:
                    G.add_edge(i, j, weight=cosine_similarity)
                    # G.add_edge(j, i, weight=combined_score)  # Add reverse edge for undirected similarity
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

if __name__ == "__main__":
    print("Loading chunk data and embeddings...")
    # take in arg of model name
    parser = argparse.ArgumentParser(description="Process corpus and create embeddings")
    parser.add_argument("--model", choices=["bert", "openai-gpt"], default="bert", help="Choose the model for embeddings")
    args = parser.parse_args()

    chunk_data = np.load(f"chunks_data_{args.model}.npy", allow_pickle=True)
    chunk_embeddings = np.load(f"chunks_embeddings_{args.model}.npy")

    # get args for similarity threshold and cosine weight
    similarity_threshold = 0.85

    print("Creating the graph...")
    G = create_embedding_graph(chunk_data, chunk_embeddings, similarity_threshold)

    print("Saving the graph...")
    save_graph(G, f"embedding_graph_{args.model}.gpickle")

    print(f"Graph created and saved as embedding_graph_{args.model}.gpickle")

    # To load the graph later:
    # loaded_G = load_graph("embedding_graph.gpickle")