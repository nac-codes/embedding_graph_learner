import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import json
import os
from time import sleep
import argparse
import pickle
from collections import deque

try:
    import torch
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


tokenizer = None
model = None
if BERT_AVAILABLE:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    


client = None
if OPENAI_AVAILABLE:
    client = OpenAI()
    try:
        client.models.list()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        OPENAI_AVAILABLE = False
        client = None


def get_embedding(text, model_name, tokenizer=None, model=None, client=None):
    if model_name == "bert":
        if not BERT_AVAILABLE:
            raise ImportError("BERT model is not available. Please install transformers and torch.")
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()
    elif model_name == "openai-gpt":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI GPT is not available. Please install openai.")
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
    else:
        raise ValueError("Invalid model name")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_openai(prompt, model_name="gpt-4", assistant_instructions="You are a helpful assistant."):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": assistant_instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def save_progress(graph_filename, visited, context, silent=False):
    save_filename = f"{os.path.splitext(graph_filename)[0]}_learn_progress.json"
    data = {
        "visited": list(visited),
        "context": list(context)
    }
    with open(save_filename, 'w') as f:
        json.dump(data, f)
    if not silent:
        print(f"Progress saved to {save_filename}")

def load_progress(graph_filename):
    save_filename = f"{os.path.splitext(graph_filename)[0]}_learn_progress.json"
    if os.path.exists(save_filename):
        with open(save_filename, 'r') as f:
            data = json.load(f)
        print(f"Progress loaded from {save_filename}")
        return set(data["visited"]), deque(data["context"], maxlen=5)
    return set(), deque(maxlen=5)

def find_most_similar_node(G, query, unvisited, model_name="bert"):
    query_embedding = get_embedding(query, model_name, client=client)
    similarities = []
    for node in unvisited:
        content_embedding = G.nodes[node]['embedding']
        if query_embedding is None:
            print("Error: Query Embedding is None")
            exit()
        if content_embedding is None:
            print("Error: Content Embedding is None")
            exit()

        similarity = cosine_similarity(query_embedding, content_embedding)
        similarities.append((node, similarity))
    return max(similarities, key=lambda x: x[1])[0]

def learn_mode(G, graph_filename, visited=None, context=None, model_name="bert"):
    if visited is None:
        visited = set()
    if context is None:
        context = deque(maxlen=5)
    
    current_node = None

    def explore_node(node):
        visited.add(node)
        print(f"\nExploring node {node}")

        content = G.nodes[node]['content']
        context_str = "\n".join(context)

        print(f"\nContent: {content}")

        prompt = f"""
        The author of the content is Haywood S. Hansell, Major General USAF. The book the content is from is The Strategic Air War against Germany and Japan.

        Previous context:
        {context_str}

        Current content:
        {content}

        Give context for the quote. Define unusual terms/events/places/people. Be detailed in your explanation and in connecting it to the previous context or other things of note.
        """

        if input("Do you want to continue with the explanation? (y/n): ").lower() != 'n':
            explanation = query_openai(prompt)
            print("\nExplanation:")
            print(explanation)
        else:
            explanation = ""

        context.append(f"Node {node}: {content}\nExplanation: {explanation}")
        save_progress(graph_filename, visited, context, silent=True)
        return node

    while len(visited) < len(G.nodes()):
        # Display progress
        print(f"\nProgress:")
        print(f"[{'=' * int(len(visited) / len(G.nodes()) * 20)}{' ' * (20 - int(len(visited) / len(G.nodes()) * 20))}] {len(visited)}/{len(G.nodes())}")

        unvisited = set(G.nodes()) - visited
        if not unvisited:
            break
        
        if not current_node:
            current_node = random.choice(list(unvisited))

        print(f"\nCurrent node: {current_node}")
        user_input = input("\nHit enter to continue with this line of conversation or type what you want to learn about next. Or type exit to save and quit: ")
        
        if user_input.lower() == 'exit':
            save_progress(graph_filename, visited, context)
            break

        if user_input.strip():
            # query open ai to make the user input into a keyword query
            prompt = f"Generate a query for a cosine similarity search from the following text: {user_input}"
            user_input = query_openai(prompt)
            print("\nKeyword query:", user_input)

            next_node = find_most_similar_node(G, user_input, unvisited, model_name)
            print(f"\nFound most relevant node based on your input.")
        else:
            # set next node to be a random child of the current node
            # if there are no children, set next node to be a random unvisited node
            next_node = random.choice(list(G.neighbors(current_node))) if G.neighbors(current_node) else random.choice(list(unvisited))
            print(f"\nContinuing with the current exploration path.")

        current_node = explore_node(next_node)

        

    print("\nLearning session complete!")
    print(f"Total nodes explored: {len(visited)}")
    print(f"Total nodes in graph: {len(G.nodes())}")

def main():
    parser = argparse.ArgumentParser(description="Graph Walk Learn Mode")
    parser.add_argument("graph_file", help="Filename of the pickled graph")
    args = parser.parse_args()

    print(f"Loading the graph from {args.graph_file}...")
    with open(args.graph_file, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    visited, context = load_progress(args.graph_file)
    print("\nStarting learn mode...")
    #strip the graph file for the model name so it's embedding_graph_{model name}.gpickle
    model_name = args.graph_file.split("_")[2].split(".")[0]
    learn_mode(G, args.graph_file, visited, context, model_name)

    if len(visited) == len(G.nodes()):
        print("\nSuccessfully explored all nodes in the graph!")
    else:
        print(f"\nNote: Only explored {len(visited)} out of {len(G.nodes())} nodes.")

    print("\nDone!")

if __name__ == "__main__":
    main()