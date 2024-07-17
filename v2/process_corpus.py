import numpy as np
import argparse
from tqdm import tqdm

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
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error when processing text: {text[:50]}...")
            print(f"API returned an error: {e}")
            return None
    else:
        raise ValueError("Invalid model name")

def process_corpus(file_path, model_name, chunk_size, overlap_ratio, tokenizer=None, model=None, client=None):
    chunk_data = []
    chunk_embeddings = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Calculate overlap size
    overlap_size = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap_size

    # Split content into overlapping chunks
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), step_size)]
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
        chunk_data.append({
            'id': i,
            'content': chunk,
            'start_index': i * step_size,
            'end_index': min(i * step_size + chunk_size, len(content))
        })
        chunk_embeddings.append(get_embedding(chunk, model_name, tokenizer, model, client))

    return chunk_data, np.array(chunk_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Process corpus and create embeddings")
    parser.add_argument("--model", choices=["bert", "openai-gpt"], default="bert", help="Choose the model for embeddings")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks (0.0 to 1.0)")
    parser.add_argument("--file-path", default="corpus.txt", help="Path to the input corpus file: default is 'corpus.txt'")
    args = parser.parse_args()

    model_name = args.model
    chunk_size = args.chunk_size
    overlap_ratio = args.overlap_ratio
    file_path = args.file_path
    file_name = file_path.split('/')[-1]
    chunk_data_file = f'chunks_data_{model_name}_{file_name}.npy'
    chunk_embeddings_file = f'chunks_embeddings_{model_name}_{file_name}.npy'

    tokenizer = None
    model = None
    client = None

    if model_name == "bert":
        if not BERT_AVAILABLE:
            print("Error: BERT model is not available. Please install transformers and torch.")
            return
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == "openai-gpt":
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI GPT is not available. Please install openai.")
            return
        client = OpenAI()

    print("Processing corpus and creating chunk data and embeddings...")
    try:
        chunk_data, chunk_embeddings = process_corpus(file_path, model_name, chunk_size, overlap_ratio, tokenizer, model, client)
    except ImportError as e:
        print(f"Error: {str(e)}")
        return
    
    print("Saving chunk data and embeddings...")
    np.save(chunk_data_file, chunk_data)
    np.save(chunk_embeddings_file, chunk_embeddings)
    
    print(f"Processing complete. Data saved to {chunk_data_file} and {chunk_embeddings_file}")

if __name__ == "__main__":
    main()