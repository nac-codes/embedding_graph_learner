import networkx as nx
import random
from tqdm import tqdm
import openai
import numpy as np
import json
import os
from time import sleep
import argparse
import pickle
from collections import deque

# Initialize OpenAI client
client = openai.OpenAI()

def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

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

def save_progress(graph_filename, threshold, mode, visited, total_questions, correct_answers, context, silent=False):
    save_filename = f"{os.path.splitext(graph_filename)[0]}_{threshold:.2f}_{mode}_progress.json"
    data = {
        "visited": list(visited),
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "context": list(context)
    }
    with open(save_filename, 'w') as f:
        json.dump(data, f)
    if not silent:
        print(f"Progress saved to {save_filename}")

def load_progress(graph_filename, threshold, mode):
    save_filename = f"{os.path.splitext(graph_filename)[0]}_{threshold:.2f}_{mode}_progress.json"
    if os.path.exists(save_filename):
        with open(save_filename, 'r') as f:
            data = json.load(f)
        print(f"Progress loaded from {save_filename}")
        return set(data["visited"]), data["total_questions"], data["correct_answers"], deque(data["context"], maxlen=5)
    return set(), 0, 0, deque(maxlen=5)

def random_walk_quiz(G, threshold, graph_filename, visited=None, total_questions=0, correct_answers=0):
    if visited is None:
        visited = set()
    path = []
    stack = []

    def dfs(node, depth=0):
        nonlocal total_questions, correct_answers
        visited.add(node)
        path.append(node)
        print(f"\n{'  ' * depth}Visiting node {node}")

        content = G.nodes[node]['content']
        question = query_openai(f"Generate an academic short answer question from the following content. Only return the question:\n\n{content}")
        print(f"\nQuestion: {question}")

        user_answer = input("Your answer (type 'exit' to save and quit): ")
        if user_answer.lower() == 'exit':
            raise KeyboardInterrupt

        total_questions += 1

        content_embedding = get_embedding(content)
        answer_embedding = get_embedding(user_answer)
        similarity = cosine_similarity(content_embedding, answer_embedding)

        is_correct = similarity > threshold
        print(f"Similarity: {similarity:.2f}")

        feedback_prompt = f"""
        Content: {content}
        Question: {question}
        User's Answer: {user_answer}
        Cosine Similarity of user's answer to the original content: {similarity}
        Correct: {"Yes" if is_correct else "No"}

        Please provide me feedback on why I got this answer {"correct" if is_correct else "wrong"} and how it could be improved based on the content.\n
        {f"If the answer was evaluated to correct but you think it is actually wrong because of some significant factual error end your feedback with a newline and: 'OVERRULED'. If it is factually accurate return a newline and: 'FACTUALLY ACCURATE'" if is_correct else ""}
        """
        feedback = query_openai(feedback_prompt)

        if "OVERRULED" in feedback.split('\n')[-1]:
            is_correct = not is_correct
            print("---The automatic evaluation has been overruled.---")
            feedback = feedback.replace("OVERRULED", "").strip()
        else:
            feedback = feedback.replace("FACTUALLY ACCURATE", "").strip()

        if is_correct:
            correct_answers += 1
        print("Correct" if is_correct else "Wrong")
        sleep(1)

        if not is_correct:
            print("Content: ", content)

        print(f"\nFeedback: {feedback}")

        progress = correct_answers / total_questions
        print("\nProgress:")
        print(f"[{'=' * int(progress * 20)}{' ' * (20 - int(progress * 20))}] {progress:.0%}")
        print(f"Correct answers: {correct_answers}/{total_questions}")

        save_progress(graph_filename, threshold, "quiz", visited, total_questions, correct_answers, [])

        neighbors = list(G.neighbors(node))
        unvisited = [n for n in neighbors if n not in visited]

        while unvisited:
            next_node = random.choice(unvisited)
            print(f"{'  ' * depth}Moving from node {node} to {next_node}")
            stack.append(node)
            dfs(next_node, depth + 1)
            unvisited = [n for n in neighbors if n not in visited]

        if stack:
            parent = stack.pop()
            print(f"{'  ' * depth}Backtracking from node {node} to {parent}")
            path.append(parent)

    try:
        while len(visited) < len(G.nodes()):
            unvisited_nodes = set(G.nodes()) - visited
            next_start = random.choice(list(unvisited_nodes))
            print(f"\nStarting new walk from {'disconnected ' if visited else ''}node {next_start}")
            dfs(next_start)

    except KeyboardInterrupt:
        print("\nQuiz interrupted by user. Progress saved.")

    return path, len(visited), total_questions, correct_answers

def learn_mode(G, graph_filename, visited=None, context=None):
    if visited is None:
        visited = set()
    if context is None:
        context = deque(maxlen=5)
    path = []
    stack = []

    def dfs(node, depth=0):
        visited.add(node)
        path.append(node)
        print(f"\n{'  ' * depth}Exploring node {node}")

        content = G.nodes[node]['content']
        context_str = "\n".join(context)

        print(f"\nContent: {content}")

        
        prompt = f"""
        The author of the content is Haywood S. Hansell, Major General USAF. The book the content is from is The Strategic Air War against Germany and Japan.

        Previous context:
        {context_str}

        Current content:
        {content}

        Give context for the quote, be detailed in your explanationa and in connecting it to the previous context or other things of note.
        """

        explanation = query_openai(prompt)
        print("\nExplanation:")
        print(explanation)

        context.append(f"Node {node}: {content}\nExplanation: {explanation}")
        save_progress(graph_filename, 0, "learn", visited, 0, 0, context, silent=True)

        user_input = input("\nPress Enter to continue, or type 'exit' to save and quit: ")
        if user_input.lower() == 'exit':
            raise KeyboardInterrupt

        neighbors = list(G.neighbors(node))
        unvisited = [n for n in neighbors if n not in visited]

        while unvisited:
            next_node = random.choice(unvisited)
            print(f"{'  ' * depth}Moving from node {node} to {next_node}")
            stack.append(node)
            dfs(next_node, depth + 1)
            unvisited = [n for n in neighbors if n not in visited]

        if stack:
            parent = stack.pop()
            print(f"{'  ' * depth}Backtracking from node {node} to {parent}")
            path.append(parent)

    try:
        while len(visited) < len(G.nodes()):
            unvisited_nodes = set(G.nodes()) - visited
            next_start = random.choice(list(unvisited_nodes))
            print(f"\nStarting new exploration from {'disconnected ' if visited else ''}node {next_start}")
            dfs(next_start)

    except KeyboardInterrupt:
        print("\nLearning session interrupted by user. Progress saved.")

    return path, len(visited)

def main():
    parser = argparse.ArgumentParser(description="Graph Walk Quiz and Learn")
    parser.add_argument("graph_file", help="Filename of the pickled graph")
    parser.add_argument("--mode", choices=["quiz", "learn"], default="quiz", help="Mode of operation: quiz or learn")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold for correct answers in quiz mode")
    args = parser.parse_args()

    print(f"Loading the graph from {args.graph_file}...")
    with open(args.graph_file, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

    if args.mode == "quiz":
        visited, total_questions, correct_answers, _ = load_progress(args.graph_file, args.threshold, "quiz")
        print(f"\nStarting quiz mode with similarity threshold {args.threshold}...")
        path, nodes_visited, total_questions, correct_answers = random_walk_quiz(
            G, args.threshold, args.graph_file, visited, total_questions, correct_answers
        )
        print("\nQuiz complete!")
        print(f"Total nodes visited: {nodes_visited}")
        print(f"Total nodes in graph: {len(G.nodes())}")
        print(f"Total questions asked: {total_questions}")
        print(f"Correct answers: {correct_answers}")
        final_grade = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        print(f"\nFinal Grade: {final_grade:.2f}%")
    else:
        visited, _, _, context = load_progress(args.graph_file, 0, "learn")
        print("\nStarting learn mode...")
        path, nodes_visited = learn_mode(G, args.graph_file, visited, context)
        print("\nLearning session complete!")
        print(f"Total nodes explored: {nodes_visited}")
        print(f"Total nodes in graph: {len(G.nodes())}")

    if nodes_visited == len(G.nodes()):
        print("\nSuccessfully visited all nodes in the graph!")
    else:
        print(f"\nNote: Only visited {nodes_visited} out of {len(G.nodes())} nodes.")

    print("\nDone!")

if __name__ == "__main__":
    main()