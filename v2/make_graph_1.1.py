import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
import re
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from rapidfuzz import fuzz

def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def get_ngrams(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def multi_ngram_search(query, text, max_n=3):
    query = normalize_text(query)
    text = normalize_text(text)
    
    query_ngrams = [get_ngrams(query, i) for i in range(1, max_n+1)]
    text_ngrams = [get_ngrams(text, i) for i in range(1, max_n+1)]
    
    score = 0
    max_score = 0
    
    for n in range(1, max_n+1):
        for q_gram in query_ngrams[n-1]:
            best_match = max((fuzz.ratio(q_gram, t_gram) for t_gram in text_ngrams[n-1]), default=0)
            score += best_match * n * n  # Weight longer n-grams more
        max_score += 100 * len(query_ngrams[n-1]) * n * n
    
    return score / max_score if max_score > 0 else 0.0

def cluster_embeddings(chunk_embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(chunk_embeddings)
    return cluster_labels, kmeans.cluster_centers_

def calculate_cluster_thresholds(chunk_embeddings, cluster_labels, cluster_centers, percentile=75):
    thresholds = {}
    for cluster in np.unique(cluster_labels):
        cluster_points = chunk_embeddings[cluster_labels == cluster]
        intra_cluster_distances = pairwise_distances(cluster_points, [cluster_centers[cluster]], metric='cosine')
        thresholds[cluster] = np.percentile(intra_cluster_distances, percentile)  # 75th percentile as threshold
    return thresholds

def create_embedding_graph_clustered(chunk_data, chunk_embeddings, cosine_weight=0.5, n_clusters=10, percentile=75):
    G = nx.DiGraph()
    edge_count = 0

    # Cluster the embeddings
    print("Clustering embeddings...")
    cluster_labels, cluster_centers = cluster_embeddings(chunk_embeddings, n_clusters)
    cluster_thresholds = calculate_cluster_thresholds(chunk_embeddings, cluster_labels, cluster_centers, percentile)
    print("Cluster thresholds:", cluster_thresholds)

    
    for i, chunk in enumerate(chunk_data):
        G.add_node(i, content=chunk['content'], cluster=cluster_labels[i])

    total_comparisons = len(chunk_data) * (len(chunk_data) - 1) // 2
    with tqdm(total=total_comparisons, desc="Comparing chunks", unit="comparison") as pbar:
        for i in range(len(chunk_data)):
            for j in range(i+1, len(chunk_data)):
                cosine_similarity = 1 - cosine(chunk_embeddings[i], chunk_embeddings[j])
                ngram_score = multi_ngram_search(chunk_data[i]['content'], chunk_data[j]['content'])
                combined_score = (cosine_similarity * cosine_weight) + (ngram_score * (1 - cosine_weight))

                # Determine threshold based on whether chunks are in the same cluster
                if cluster_labels[i] == cluster_labels[j]:
                    threshold = cluster_thresholds[cluster_labels[i]]
                else:
                    threshold = (cluster_thresholds[cluster_labels[i]] + cluster_thresholds[cluster_labels[j]]) / 2

                if combined_score > threshold:
                    G.add_edge(i, j, weight=combined_score)
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
    chunk_data = np.load("chunks_data_bert.npy", allow_pickle=True)
    chunk_embeddings = np.load("chunks_embeddings_bert.npy")

    cosine_weight = 0.7
    n_clusters = 6  # You can adjust this based on your dataset
    percentile = 95  # You can adjust this based on your dataset

    print("Creating the graph...")
    G = create_embedding_graph_clustered(chunk_data, chunk_embeddings, cosine_weight, n_clusters, percentile)

    print("Saving the graph...")
    save_graph(G, "embedding_graph_clustered.gpickle")

    print("Graph created and saved as embedding_graph_clustered.gpickle")