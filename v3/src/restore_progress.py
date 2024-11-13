import json
import pickle
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import argparse
import os

# Initialize OpenAI client
client = OpenAI()

def get_embedding(text, client=None):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        if response.data and response.data[0].embedding:
            return np.array(response.data[0].embedding)
        else:
            print("OpenAI API returned an unexpected response structure")
            return None
    except Exception as e:
        print(f"Error when processing text: {text[:50]}...")
        print(f"API returned an error: {e}")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar_node(content, G):
    content_embedding = get_embedding(content, client)
    if content_embedding is None:
        return None

    max_similarity = -1
    most_similar_node = None

    for node, data in G.nodes(data=True):
        node_content = data.get('content', '')
        node_embedding = data.get('embedding')
        
        if node_embedding is None:
            print(f"Node {node} has no embedding, calculating...")
            node_embedding = get_embedding(node_content, client)
            G.nodes[node]['embedding'] = node_embedding

        if node_embedding is not None:
            similarity = cosine_similarity(content_embedding, node_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_node = node

    return most_similar_node

def restore_progress(graph_file, notes_file, progress_file):
    # Load the graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    # Load the notes
    with open(notes_file, 'r') as f:
        notes = json.load(f)

    # Load the progress
    with open(progress_file, 'r') as f:
        progress = json.load(f)

    visited = set(progress.get("visited", []))

    # Process each note
    for note in tqdm(notes, desc="Processing notes"):
        content = note.get('content', '')
        print(f"Processing note: {content[:100]}...")
        if content:
            most_similar_node = find_most_similar_node(content, G)
            if most_similar_node is not None:
                print(f"Most similar node: {most_similar_node}")
                print(f"Node content: {G.nodes[most_similar_node]['content'][:100]}...")
                G.nodes[most_similar_node]['notes'] = note.get('note', '')
                G.nodes[most_similar_node]['ai_explanation'] = note.get('ai_explanation', '')
                visited.add(most_similar_node)

    # Update progress
    progress["visited"] = list(visited)

    # Save updated graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)

    # Save updated progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

    print(f"Restored progress: {len(visited)} nodes visited")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore progress from notes file")
    parser.add_argument("graph_file", help="Path to the .gpickle graph file")
    parser.add_argument("notes_file", help="Path to the notes JSON file")
    parser.add_argument("progress_file", help="Path to the progress JSON file")
    args = parser.parse_args()

    restore_progress(args.graph_file, args.notes_file, args.progress_file)