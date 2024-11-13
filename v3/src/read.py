import networkx as nx
import random
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import pickle
from collections import deque
import signal
import sys
import readline  # For better input handling
from openai import OpenAI
import readline



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

def query_openai(prompt, model_name="gpt-3.5-turbo", assistant_instructions="You are a helpful assistant."):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": assistant_instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def multiline_input(prompt):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        lines.append(line)
    return "\n".join(lines).strip()

def save_progress(graph_file_path, G, visited, context, silent=False):
    # Save progress in the same directory as the graph file
    graph_dir = os.path.dirname(graph_file_path)
    graph_filename = os.path.basename(graph_file_path)
    save_filename = os.path.join(graph_dir, f"{os.path.splitext(graph_filename)[0]}_learn_progress.json")
    # Convert numpy types to Python native types for JSON serialization
    visited_list = [int(node) for node in visited]
    context_list = list(context)
    data = {
        "visited": visited_list,
        "context": context_list
    }
    try:
        with open(save_filename, 'w') as f:
            json.dump(data, f)
        if not silent:
            print(f"\nProgress saved to {save_filename}")
    except Exception as e:
        print(f"Error saving progress: {e}")

    # Save updated graph with notes and AI explanations
    try:
        with open(graph_file_path, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        if not silent:
            print(f"Graph saved to {graph_file_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")

    # Save notes to notes.json
    notes = []
    for node, data in G.nodes(data=True):
        note = data.get('notes', '').strip()
        ai_explanation = data.get('ai_explanation', '').strip()
        if (note or ai_explanation) and (note != "" or ai_explanation != ""):
            # Attempt to load metadata for this node
            author = data.get('author', 'Unknown')
            title = data.get('title', 'Unknown')
            date = data.get('publication_date', 'Unknown')
            # pages = data.get('pages', 'Unknown')
            content = data.get('content', '')
            chunk_file = data.get('chunk_file', '')

            notes.append({
                "node": int(node),  # Convert numpy.int64 to int
                "file": data['file'],
                # "pages": pages,
                "author": author,
                "title": title,
                "publication_date": date,
                "content": content,
                "ai_explanation": ai_explanation,
                "note": note,
                "chunk_file": chunk_file
            })
    if notes:
        notes_filename = os.path.join(graph_dir, f"{os.path.splitext(graph_filename)[0]}_notes.json")
        print(f"Saving {len(notes)} notes to {notes_filename}")
        try:
            with open(notes_filename, 'w') as f:
                json.dump(notes, f, indent=4)
            if not silent:
                print(f"Notes saved to {notes_filename}")
        except Exception as e:
            print(f"Error saving notes: {e}")

def load_progress(graph_file_path):
    # Load progress from the same directory as the graph file
    graph_dir = os.path.dirname(graph_file_path)
    graph_filename = os.path.basename(graph_file_path)
    save_filename = os.path.join(graph_dir, f"{os.path.splitext(graph_filename)[0]}_learn_progress.json")
    if os.path.exists(save_filename):
        try:
            with open(save_filename, 'r') as f:
                data = json.load(f)
            visited = set(data.get("visited", []))
            context = deque(data.get("context", []), maxlen=5)
            print(f"Progress loaded from {save_filename}")
            return visited, context
        except json.JSONDecodeError as e:
            print(f"Error loading progress file {save_filename}: {e}")
            print("Starting with empty progress.")
            return set(), deque(maxlen=5)
    return set(), deque(maxlen=5)

def find_most_similar_nodes(G, query, unvisited, client, top_k=15):
    selected_authors = select_authors(G, query)

    query_embedding = get_embedding(query, client)
    if query_embedding is None:
        return random.sample(list(unvisited), min(top_k, len(unvisited)))

    similarities = []
    for node in unvisited:
        author = G.nodes[node].get('author', 'Unknown')
        title = G.nodes[node].get('title', 'Unknown')
        
        if selected_authors and author not in selected_authors:
            continue

        content_embedding = G.nodes[node].get('embedding', None)
        if content_embedding is None:
            print(f"Node {node} has no embedding")
            continue
        
        similarity = cosine_similarity(query_embedding, content_embedding)
        content = G.nodes[node]['content']
        similarities.append((node, similarity, content, author, title))

    if not similarities:
        print("No matching nodes found. Falling back to random selection.")
        return random.sample(list(unvisited), min(top_k, len(unvisited)))

    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Number of similarities calculated: {len(similarities)}")
    print(f"Top {top_k} similarities:")
    for node, sim, _, author, title in top_similarities:
        print(f"Node: {node}, Similarity: {sim:.4f}, Title: {title}, Author: {author}")

    return top_similarities

def select_top_quotes(query, top_similarities, client):
    prompt = f"""Given the following user query and 15 potentially relevant text passages, 
    select the 5 most relevant passages for a literature review. 
    Prioritize both relevance to the query and diversity of opinion (different authors) if possible.
    Explain your selection briefly, focusing on how each selected passage contributes to a comprehensive review.
    Return the selected nodes in the following JSON format, NODE IDS:
    {{
        "selected_nodes": [<list of 5 selected node ids>]
    }}

    User query: {query}

    Passages:
    """
    for i, (node, sim, content, author, title) in enumerate(top_similarities, 1):
        prompt += f"\nNode {node}: {title} by {author}"
        # get the middle 250 characters of the content
        prompt += f"\n{content[len(content)//2-250:len(content)//2+250]}"

    print(prompt)

    response = query_openai(prompt, model_name="gpt-4o-mini")
    print("GPT Quote Selection Response:", response)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        response_json = json.loads(response[json_start:json_end])
        selected_nodes = response_json.get("selected_nodes", [])
        return [node for node, _, _, _, _ in top_similarities if node in selected_nodes]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing GPT response for quote selection: {e}")
        return [node for node, _, _, _, _ in top_similarities[:5]]

# def select_authors_and_titles(G, query, client):
#     # Get unique authors and titles
#     authors = set(data.get('author', 'Unknown') for _, data in G.nodes(data=True))
#     titles = set(data.get('title', 'Unknown') for _, data in G.nodes(data=True))

#     # Prepare prompt for GPT
#     prompt = f"""Given the following user query and lists of authors and titles, 
#     determine which authors and/or titles (if any) the user is likely interested in.
#     If the query doesn't specify any particular authors or titles, return an empty list for both.
#     Return your selection in the following JSON format:
#     {{
#         "authors": [<list of selected authors>],
#         "titles": [<list of selected titles>]
#     }}

#     User query: {query}

#     Authors:
#     {', '.join(authors)}

#     Titles:
#     {', '.join(titles)}
#     """

#     # Get GPT's selection
#     print(f"Author/Title Prompt: {prompt}")
#     response = query_openai(prompt, model_name="gpt-4o-mini")
#     print("GPT Author/Title Selection Response:", response)

#     try:
#         # Extract JSON from response
#         json_start = response.find("{")
#         json_end = response.rfind("}") + 1
#         response_json = json.loads(response[json_start:json_end])
        
#         selected_authors = response_json.get("authors", [])
#         selected_titles = response_json.get("titles", [])

#         return selected_authors, selected_titles
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error parsing GPT response for author/title selection: {e}")
#         return [], []

def find_most_similar_node(G, query, unvisited, client, top_k=5):
    # Use the new function to get selected authors and titles
    selected_authors = select_authors(G, query)

    query_embedding = get_embedding(query, client)
    if query_embedding is None:
        return random.choice(list(unvisited))

    similarities = []
    for node in unvisited:
        author = G.nodes[node].get('author', 'Unknown')        
        
        # Filter by selected authors or titles if any
        if selected_authors and author not in selected_authors:
            continue        

        content_embedding = G.nodes[node].get('embedding', None)
        if content_embedding is None:
            print(f"Node {node} has no embedding")
            continue
        
        similarity = cosine_similarity(query_embedding, content_embedding)
        content = G.nodes[node]['content']
        similarities.append((node, similarity, content, author, title))

    if not similarities:
        print("No matching nodes found. Falling back to random selection.")
        return random.choice(list(unvisited))

    # Sort by similarity and get top_k results
    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Number of similarities calculated: {len(similarities)}")
    print(f"Top {top_k} similarities:")
    for node, sim, _, author, title in top_similarities:
        print(f"Node: {node}, Similarity: {sim:.4f}, Title: {title}")

    # Prepare prompt for GPT
    prompt = f"""Given the following user query and {top_k} potentially relevant text passages, 
    select the passage that best answers the query. If none are particularly relevant, 
    select the most informative passage related to the topic. Explain the relevance of each one and then return in the following json format. Remember to return the node_id
    {{
        "node": <node_id>
    }}

    User query: {query}

    Passages:
    """
    for i, (node, sim, content, author, title) in enumerate(top_similarities, 1):
        prompt += f"\nNode {node}: {title} by {author}"
        prompt += f"\n{content[:1500]}..." # Truncate long contents

    print(prompt)

    # Get GPT's selection
    response = query_openai(prompt, model_name="gpt-4o-mini")
    print(response)
    try:
        # extract json from response find the section bookended by ```json and ```
        json_start = response.find("```json") + len("```json")
        json_end = response.find("```", json_start)
        response_json = json.loads(response[json_start:json_end])
        selected_node = response_json.get("node")

        # Find the corresponding node in top_similarities
        for node, _, _, _, _ in top_similarities:
            if node == selected_node:
                return node

        # If the selected node is not in top_similarities, fall back to the highest cosine similarity
        print(f"Selected node {selected_node} not found in top similarities. Falling back to highest similarity.")
        return top_similarities[0][0]
    except (ValueError, IndexError, json.JSONDecodeError):
        # If GPT's response is not valid, fall back to the highest cosine similarity
        print("Error parsing GPT response. Falling back to highest similarity.")
        return top_similarities[0][0]

def select_authors(G, query):
    # Get unique authors
    authors = set(data.get('author', 'Unknown') for _, data in G.nodes(data=True))
    
    # Convert query to lowercase for case-insensitive matching. only include words that are capitalized
    query_words = set(word.lower() for word in query.split() if len(word) > 2 and word[0].isupper())
    
    print(f"Authors: {authors}")
    print(f"Query words: {query_words}")
    
    # Find matching authors
    selected_authors = set()
    for author in authors:
        author_words = set(word.lower() for word in author.split() if len(word) > 2)
        if any(word in query_words for word in author_words):
            selected_authors.add(author)
    
    print(f"Selected authors: {selected_authors}")
    
    return list(selected_authors)

def load_metadata(file_path):
    """
    Load metadata from .txt.met file.
    """
    metadata_file = file_path + '.met'
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get('author', 'Unknown'), metadata.get('title', 'Unknown'), metadata.get('publication_date', 'Unknown')
        except json.JSONDecodeError as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
            return 'Unknown', 'Unknown', 'Unknown'
    else:
        return 'Unknown', 'Unknown', 'Unknown'

def learn_mode(G, graph_file_path, visited=None, context=None):
    if visited is None:
        visited = set()
    if context is None:
        context = deque(maxlen=5)

    graph_dir = os.path.dirname(graph_file_path)

    current_node = None

    def explore_node(node, user_input=None):
        nonlocal visited
        visited.add(node)
        input("Press Enter to continue...")
        # os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        content = G.nodes[node]['content']
        if len(context) > 0:
            # get the latest context item and only the last 750 characters
            context_str = context[-1][-750:]
        else:
            context_str = ""

        # Extract file info
        file_info = G.nodes[node]['file']
        pages = G.nodes[node]['pages']
        author = G.nodes[node].get('author', 'Unknown')
        title = G.nodes[node].get('title', 'Unknown')
        date = G.nodes[node].get('publication_date', 'Unknown')
        chunk_file = G.nodes[node].get('chunk_file', '')

        # Display node content with better formatting
        print(f"📖 **{title}** by *{author}* ({date})")
        # print(f"🗓️ Pages: {pages}")
        print(f"Chunk file: {chunk_file}")
        print("\n---\n")
        print(f"{content}")
        print("\n---\n")

        # Check if AI explanation already exists
        existing_ai_explanation = G.nodes[node].get('ai_explanation', '')

        # Ask if user wants an AI explanation
        print("Press [E] for AI Explanation, [N] to take a Note, [S] to Skip, or [Q] to Quit.")
        choice = input("Your choice: ").strip().lower()

        if choice == 'e':
            if existing_ai_explanation:
                print("\n📝 **AI Explanation (Previously Generated):**")
                print(f"{existing_ai_explanation}")
                input("\nPress Enter to continue...")
            else:

                if user_input:
                    prompt = f"""
You are a PhD in History. You are creating a literature review on the following topic: "{user_input}". This quote is one part of the literature review. Just provide an anlysis of this quote as it relates to the topic.
The author of the content is {author}. The book the content is from is "{title}". The publication date is {date}.

Context (the last part of the literature review you were writing):
{context_str}

Content to analyze now:
{content}

You should respond to the request in an extremely detailed manner. Provide an answer and back it up with quoted evidence from the text. Make the quotes lenghty.
"""
                else:
                    prompt = f"""
You are a PhD in History. You are reacting to the following content.
The author of the content is {author}. The book the content is from is {title}. The publication date is {date}.

Previous context:
{context_str}

Current content:
{content}

Provide an explanation of the above content, and provide further detailed context if possible. Define unusual terms, events, places, and people. Be detailed in your explanation and connect it to the previous context or other notable information.
"""
                explanation = query_openai(prompt, model_name="gpt-4o")
                print("\n📝 **AI Explanation:**")
                print(f"{explanation}")
                G.nodes[node]['ai_explanation'] = explanation  # Save AI explanation to the node
                input("\nPress Enter to continue...")

            # Option to take a note after the explanation            
            note = multiline_input("Would you like to add a note? (Press Ctrl+D to finish, leave empty to skip):")
            if note:
                G.nodes[node]['notes'] = note
                print("Note saved.")
            context.append(f"Node {node}: {content}\nExplanation: {G.nodes[node]['ai_explanation']}")

        elif choice == 'n':
            note = multiline_input("Enter your note (Press Ctrl+D to finish, leave empty to skip):")
            if note:
                G.nodes[node]['notes'] = note
                print("Note saved.")
        elif choice == 'q':
            save_progress(graph_file_path, G, visited, context)
            sys.exit(0)

        return node

    # Handle graceful exit on Ctrl+C
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Saving progress and exiting...")
        save_progress(graph_file_path, G, visited, context)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while len(visited) < len(G.nodes()):
        # Display progress bar
        progress = int(len(visited) / len(G.nodes()) * 20)
        print(f"\nProgress: [{'=' * progress}{' ' * (20 - progress)}] {len(visited)}/{len(G.nodes())}")

        unvisited = set(G.nodes()) - visited
        if not unvisited:
            break

        if not current_node:
            current_node = random.choice(list(unvisited))

        user_input = input("\nPress Enter to continue, type a topic to search, or 'exit' to quit: ").strip()

        if user_input.lower() in ['exit', 'q', 'quit']:
            save_progress(graph_file_path, G, visited, context)
            break

        if user_input:
            # Generate a keyword query using OpenAI
            prompt = f"Generate a search string for RAG based on the following request. Keep the full length of the request! it should have a similar length to the request: {user_input}. JUST RETURN THE STRING, NO OTHER TEXT."
            keyword_query = query_openai(prompt, model_name="gpt-4o-mini")
            print(f"\nSearching for: {keyword_query}")

            single_or_multiple = 1 # 0 for single, 1 for multiple
            if single_or_multiple == 0:
                next_node = find_most_similar_node(G, keyword_query, unvisited, client)
                print(f"\nSelected node: {next_node}")
                print(f"Node content: {G.nodes[next_node]['content'][:100]}...")
                print(f"\nFound a relevant section.")
                current_node = explore_node(next_node, user_input)
            else:
                # make it so that it selects from all nodes, not just unvisited
                top_similarities = find_most_similar_nodes(G, keyword_query, list(G.nodes()), client)
                selected_nodes = select_top_quotes(keyword_query, top_similarities, client)
                
                print(f"\nSelected nodes for literature review: {selected_nodes}")
                for node in selected_nodes:
                    current_node = explore_node(node, user_input)
                    if current_node is None:  # If explore_node returns None, it means the user wants to quit
                        save_progress(graph_file_path, G, visited, context)
                        return                
            
            
        else:
            # Continue to a connected node or random unvisited node
            unvisited = set(G.nodes()) - visited  # Recalculate unvisited set
            # if current_node is not None:
            #     neighbors = list(G.neighbors(current_node))
            #     unvisited_neighbors = [n for n in neighbors if n in unvisited]
            #     if unvisited_neighbors:
            #         next_node = random.choice(unvisited_neighbors)
            #         print("\nMoving to a connected section.")
            #     else:
            #         next_node = random.choice(list(unvisited))
            #         print("\nMoving to a new random section.")
            # else:
            next_node = random.choice(list(unvisited))
            print("\nStarting with a random section.")

            current_node = explore_node(next_node)
            # Add debug prints
            print(f"Visited nodes: {visited}")
            print(f"Current node: {current_node}")
            print(f"Unvisited nodes: {set(G.nodes()) - visited}")

    print("\nLearning session complete.")
    print(f"Total nodes explored: {len(visited)} out of {len(G.nodes())}")

def main():
    parser = argparse.ArgumentParser(description="Graph Walk Learn Mode with Notes")
    parser.add_argument("graph_file", help="Path to the .gpickle graph file")
    args = parser.parse_args()

    graph_file_path = args.graph_file

    if not os.path.exists(graph_file_path):
        print(f"Graph file {graph_file_path} does not exist.")
        sys.exit(1)

    print(f"Loading the graph from {graph_file_path}...")
    with open(graph_file_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # Ensure all nodes have a 'notes' and 'ai_explanation' attribute
    for node in G.nodes():
        if 'notes' not in G.nodes[node]:
            G.nodes[node]['notes'] = ''
        if 'ai_explanation' not in G.nodes[node]:
            G.nodes[node]['ai_explanation'] = ''

    visited, context = load_progress(graph_file_path)
    print("\nStarting learn mode...")

    learn_mode(G, graph_file_path, visited, context)

    if len(visited) == len(G.nodes()):
        print("\nSuccessfully explored all nodes in the graph!")
    else:
        print(f"\nNote: Explored {len(visited)} out of {len(G.nodes())} nodes.")

    print("\nDone!")

if __name__ == "__main__":
    main()
