from sys import argv
import openai
import random
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from time import sleep
import json
import os

client = openai.OpenAI()

# Download the necessary NLTK data
nltk.download('punkt')

def query_openai(prompt, model_name="gpt-4o", assistant_instructions="You are a helpful writing assistant."):
    models = client.models.list()
    model_names = [model.id for model in models.data]
    if model_name in model_names:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": assistant_instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    else:
        return "Model not found. Try again."

def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_chunks(text, chunk_size=200, overlap=50):
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def save_progress(corpus_filename, threshold, chunks, answered, total_questions):
    save_filename = f"{os.path.splitext(corpus_filename)[0]}_{threshold:.2f}_progress.json"
    data = {
        "chunks": chunks,
        "answered": answered,
        "total_questions": total_questions
    }
    with open(save_filename, 'w') as f:
        json.dump(data, f)
    print(f"Progress saved to {save_filename}")

def load_progress(corpus_filename, threshold):
    save_filename = f"{os.path.splitext(corpus_filename)[0]}_{threshold:.2f}_progress.json"
    if os.path.exists(save_filename):
        with open(save_filename, 'r') as f:
            data = json.load(f)
        print(f"Progress loaded from {save_filename}")
        return data["chunks"], data["answered"], data["total_questions"]
    return None, None, None

def main():
    threshold = float(argv[2])
    corpus_filename = str(argv[1])
    
    # Check for saved progress
    chunks, answered, total_questions = load_progress(corpus_filename, threshold)
    
    if chunks is None:
        # No saved progress, start fresh
        with open(corpus_filename, 'r') as file:
            corpus = file.read()
        chunks = create_chunks(corpus)
        answered = [0] * len(chunks)
        total_questions = 0
    
    try:
        while 0 in answered:
            unanswered_indices = [i for i, a in enumerate(answered) if a == 0]
            chunk_index = random.choice(unanswered_indices)
            chunk = chunks[chunk_index]

            question = query_openai(f"Generate an academic short answer question from the following content. Only return the question:\n\n{chunk}")
            print(f"\nQuestion: {question}")

            user_answer = input("Your answer (type 'exit' to save and quit): ")
            if user_answer.lower() == 'exit':
                raise KeyboardInterrupt
            
            total_questions += 1

            chunk_embedding = get_embedding(chunk)
            answer_embedding = get_embedding(user_answer)
            similarity = cosine_similarity(chunk_embedding, answer_embedding)

            is_correct = similarity > threshold
            print(f"Similarity: {similarity:.2f}")

            feedback_prompt = f"""
            Content: {chunk}
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
            
            answered[chunk_index] = 1 if is_correct else 0
            print("Correct" if is_correct else "Wrong")
            sleep(1)

            if not is_correct:
                print("Content: ", chunk)
            
            print(f"\nFeedback: {feedback}")

            correct_answers = sum(answered)
            progress = correct_answers / len(chunks)
            print("\nProgress:")
            print(f"[{'=' * int(progress * 20)}{' ' * (20 - int(progress * 20))}] {progress:.0%}")
            # print progress not as percent but progress / total questions
            print(f"{len(chunks) - correct_answers} questions to go!")

        print("\nWell done!")
        final_grade = (sum(answered) / total_questions) * 100
        print(f"Final Grade: {final_grade:.2f}% ({sum(answered)} correct answers out of {total_questions} questions)")

    except KeyboardInterrupt:
        print("\nExiting program. Saving progress...")
    finally:
        save_progress(corpus_filename, threshold, chunks, answered, total_questions)

if __name__ == "__main__":
    main()