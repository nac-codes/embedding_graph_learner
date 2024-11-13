import os
import re
import argparse
import numpy as np
from tqdm import tqdm
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

client = OpenAI()

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_pages_and_text(txt_file_path):
    """
    Extract text between 'Page' indicators and assign page ranges.
    Remove lines that say 'Frame %d:'.
    """
    pages = []
    current_text = ""
    current_start_page = None
    current_end_page = None

    page_pattern = re.compile(r'^Page\s+(\d+)(?:\s+of\s+\d+)?$', re.IGNORECASE)
    frame_pattern = re.compile(r'^Frame\s+\d+:$', re.IGNORECASE)

    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check for 'Page' indicator
            page_match = page_pattern.match(line)
            if page_match:
                if current_text:
                    # Save the previous page's text
                    pages.append({
                        'start_page': current_start_page,
                        'end_page': current_end_page,
                        'content': current_text.strip()
                    })
                    current_text = ""
                current_start_page = int(page_match.group(1))
                current_end_page = current_start_page  # Will update if needed
            elif frame_pattern.match(line):
                # Skip 'Frame %d:' lines
                continue
            else:
                if current_text and current_start_page is not None:
                    # Update end page if content continues
                    current_end_page = current_start_page
                current_text += line + " "

    # Add the last page's text
    if current_text:
        pages.append({
            'start_page': current_start_page,
            'end_page': current_end_page,
            'content': current_text.strip()
        })

    return pages

def create_chunks(pages, chunk_size, overlap_ratio):
    """
    Create chunks from the pages, keeping track of page numbers.
    """
    chunks = []
    current_chunk = ""
    current_pages = []
    overlap_size = int(chunk_size * overlap_ratio)

    for page in pages:
        content = page['content']
        page_range = (page['start_page'], page['end_page'])
        current_chunk += content + " "
        current_pages.append(page_range)

        while len(current_chunk) >= chunk_size:
            chunk_content = current_chunk[:chunk_size]
            chunks.append({
                'content': chunk_content.strip(),
                'pages': current_pages.copy()
            })
            current_chunk = current_chunk[chunk_size - overlap_size:]
            current_pages = [current_pages[-1]]  # Keep last page range for overlap

    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'pages': current_pages.copy()
        })

    return chunks

# def get_embedding(text, model_name="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     try:
#         response = client.embeddings.create(input=[text], model=model_name)
#         return np.array(response.data[0].embedding)
#     except Exception as e:
#         print(f"Error when processing text: {text[:50]}...")
#         print(f"API returned an error: {e}")
#         return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding_with_retry(text, model_name="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model_name, timeout=10)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error when processing text: {text[:50]}...")
        print(f"API returned an error: {e}")
        raise  # This will trigger a retry

def get_embedding(text, model_name="text-embedding-ada-002"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return get_embedding_with_retry(text, model_name)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get embedding after {max_retries} attempts.")
                return None
            else:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying

def process_txt_file(txt_file_path, output_dir, chunk_size, overlap_ratio):
    print(f"Processing {txt_file_path}...")
    pages = extract_pages_and_text(txt_file_path)
    if not pages:
        print(f"No content found in {txt_file_path}. Skipping.")
        return

    # Create chunks
    chunks = create_chunks(pages, chunk_size, overlap_ratio)

    # Generate embeddings for chunks
    chunk_embeddings = []
    for chunk in tqdm(chunks, desc="Generating embeddings", unit="chunk"):
        embedding = get_embedding(chunk['content'])
        if embedding is not None:
            chunk_embeddings.append(embedding)
        else:
            chunk_embeddings.append(np.zeros(1536))  # Assuming embedding size of 1536
        time.sleep(0.1)

    # Save chunks and embeddings
    base_filename = os.path.splitext(os.path.basename(txt_file_path))[0]
    sanitized_filename = sanitize_filename(base_filename)
    chunks_output_path = os.path.join(output_dir, f"{sanitized_filename}_chunks.json")
    embeddings_output_path = os.path.join(output_dir, f"{sanitized_filename}_embeddings.npy")

    with open(chunks_output_path, 'w') as f:
        json.dump(chunks, f, indent=4)

    np.save(embeddings_output_path, np.array(chunk_embeddings))

    print(f"Finished processing {txt_file_path}. Chunks and embeddings saved in {output_dir}.")

def process_corpora(corpora_dir, chunk_size, overlap_ratio):
    for subdir in os.listdir(corpora_dir):
        subdir_path = os.path.join(corpora_dir, subdir)
        if os.path.isdir(subdir_path):
            # check if chunks and embeddings already exist
            sanitized_filename = sanitize_filename(subdir)
            chunks_output_path = os.path.join(subdir_path, f"{sanitized_filename}_chunks.json")
            embeddings_output_path = os.path.join(subdir_path, f"{sanitized_filename}_embeddings.npy")
            if os.path.exists(chunks_output_path) and os.path.exists(embeddings_output_path):
                print(f"Skipping {subdir_path} because chunks and embeddings already exist.")
                continue

            # Look for the txt file that corresponds to the subfolder name
            txt_filename = f"{subdir}.txt"
            txt_file_path = os.path.join(subdir_path, txt_filename)
            if os.path.isfile(txt_file_path):
                output_dir = subdir_path  # Save outputs in the subfolder
                process_txt_file(txt_file_path, output_dir, chunk_size, overlap_ratio)
            else:
                print(f"No corresponding txt file {txt_filename} found in {subdir_path}. Skipping.")
        else:
            continue  # Skip files in the corpora_dir root

def main():
    parser = argparse.ArgumentParser(description="Process text files and create embeddings")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Size of text chunks")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks (0.0 to 1.0)")
    parser.add_argument("--corpora-dir", required=True, help="Directory containing corpora subfolders")
    args = parser.parse_args()

    corpora_dir = args.corpora_dir

    if not os.path.isdir(corpora_dir):
        print(f"The directory {corpora_dir} does not exist.")
        return

    process_corpora(corpora_dir, args.chunk_size, args.overlap_ratio)

    print("Processing complete for all text files.")

if __name__ == "__main__":
    main()
