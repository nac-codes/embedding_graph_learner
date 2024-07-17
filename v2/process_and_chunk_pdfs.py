import os
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import json
import re
import argparse
import numpy as np
from openai import OpenAI

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Initialize OpenAI client
client = OpenAI()

def query_openai(prompt, model_name="gpt-4", assistant_instructions="You are a helpful assistant."):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": assistant_instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def process_pdf_page(page, page_num, perform_ocr=False):
    """Process a single PDF page, either by extracting text or performing OCR."""
    if perform_ocr:
        image = page.convert('RGB')
        text = pytesseract.image_to_string(image)
    else:
        text = page.extract_text()
    
    return {
        'page_number': page_num,
        'content': text.strip() if text else ""
    }

def extract_text_from_pdf(pdf_path, pdf_filename):
    """Extract text from a PDF, combining regular text extraction and OCR as needed."""
    pages = []
    with open(pdf_path, 'rb') as file_in:
        reader = PyPDF2.PdfReader(file_in)
        pdf_images = convert_from_path(pdf_path, 300)
        
        for page_num, (pdf_page, page_image) in enumerate(tqdm(zip(reader.pages, pdf_images), total=len(reader.pages)), 1):
            # Try regular text extraction first
            page_data = process_pdf_page(pdf_page, page_num)
            
            # If no text was extracted, perform OCR
            if not page_data['content']:
                page_data = process_pdf_page(page_image, page_num, perform_ocr=True)
            
            pages.append(page_data)
            
    return pages

def extract_metadata(text, filename):
    prompt = f"""
    Based on the following text extracted from a PDF and its filename, please extract the author name, title, and publication date. If you can't find a specific piece of information, use 'Unknown' for that field.

    Filename: {filename}

    Extracted text:
    {text[:2000]}  # Using the first 2000 characters for brevity

    Please respond in JSON format with the following structure:
    {{
        "author": "Author name",
        "title": "Document title",
        "publication_date": "YYYY-MM-DD or Unknown"
    }}
    """
    
    response = query_openai(prompt)
    response = response.replace("```json", "").replace("```", "")
    try:
        metadata = json.loads(response)
        return metadata
    except json.JSONDecodeError:
        print(f"Error parsing OpenAI response for {filename}. Using default values.")
        print("Response:", response)
        return {
            "author": "Unknown",
            "title": "Unknown",
            "publication_date": "Unknown"
        }

def get_user_input_for_unknown(metadata):
    for key, value in metadata.items():
        if value == "Unknown":
            while True:
                user_input = input(f"Please enter the {key} (or press Enter to keep it as Unknown): ").strip()
                if user_input:
                    metadata[key] = user_input
                    break
                else:
                    confirm = input("Are you sure you want to keep it as Unknown? (y/n): ").lower()
                    if confirm == 'y':
                        break
    return metadata

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def create_chunks(pages, chunk_size, overlap_ratio):
    chunks = []
    current_chunk = ""
    current_chunk_pages = []
    overlap_size = int(chunk_size * overlap_ratio)

    for page in pages:
        current_chunk += page['content'] + " "
        current_chunk_pages.append(page['page_number'])

        while len(current_chunk) >= chunk_size:
            chunk_content = current_chunk[:chunk_size]
            chunks.append({
                'content': chunk_content,
                'pages': current_chunk_pages.copy(),
                'start_index': len(chunks) * (chunk_size - overlap_size),
                'end_index': len(chunks) * (chunk_size - overlap_size) + chunk_size
            })
            current_chunk = current_chunk[chunk_size - overlap_size:]
            current_chunk_pages = [current_chunk_pages[-1]]

    if current_chunk:
        chunks.append({
            'content': current_chunk,
            'pages': current_chunk_pages,
            'start_index': len(chunks) * (chunk_size - overlap_size),
            'end_index': len(chunks) * (chunk_size - overlap_size) + len(current_chunk)
        })

    return chunks

def get_embedding(text, model_name="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model_name)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error when processing text: {text[:50]}...")
        print(f"API returned an error: {e}")
        return None

def process_pdf(pdf_filename, chunk_size, overlap_ratio):
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    
    print(f"Processing {pdf_filename}...")
    pages = extract_text_from_pdf(pdf_path, pdf_filename)
    
    full_text = " ".join([page['content'] for page in pages])
    
    # Extract metadata using OpenAI
    metadata = extract_metadata(full_text, pdf_filename)
    
    # Get user input for unknown metadata
    metadata = get_user_input_for_unknown(metadata)
    
    # Create the output filename based on metadata
    output_filename = f"{sanitize_filename(metadata['author'])}_{sanitize_filename(metadata['title'])}_{sanitize_filename(metadata['publication_date'])}"
    
    # Save the full text to a file
    full_text_path = os.path.join(pdf_dir, f"{output_filename}_full.txt")
    with open(full_text_path, 'w') as file_out:
        for page in pages:
            file_out.write(f"Page {page['page_number']}:\n{page['content']}\n\n")
    
    # Create chunks
    chunks = create_chunks(pages, chunk_size, overlap_ratio)
    
    # Generate embeddings for chunks
    chunk_embeddings = []
    for chunk in tqdm(chunks, desc="Generating embeddings", unit="chunk"):
        embedding = get_embedding(chunk['content'])
        if embedding is not None:
            chunk_embeddings.append(embedding)
    
    # make directory chunks in pdf_dir
    os.makedirs(os.path.join(pdf_dir, "chunks"), exist_ok=True)

    # Save chunk data and embeddings
    chunk_data_file = f'chunks/{output_filename}_chunks.json'
    chunk_embeddings_file = f'chunks/{output_filename}_embeddings.npy'
    
    with open(os.path.join(pdf_dir, chunk_data_file), 'w') as f:
        json.dump(chunks, f, indent=4)
    
    np.save(os.path.join(pdf_dir, chunk_embeddings_file), np.array(chunk_embeddings))
    
    # Save metadata to a JSON file
    metadata_file = f"{output_filename}_metadata.json"
    metadata_path = os.path.join(pdf_dir, metadata_file)
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    
    print(f"Processed {pdf_filename}. Saved full text, chunks, embeddings, and metadata.")

def main():
    parser = argparse.ArgumentParser(description="Process PDF and create embeddings")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks (0.0 to 1.0)")
    parser.add_argument("--pdf-dir", default=".", help="Directory containing PDF files")
    args = parser.parse_args()

    global pdf_dir
    pdf_dir = args.pdf_dir

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            process_pdf(pdf_file, args.chunk_size, args.overlap_ratio)

    print("Processing complete for all PDFs.")

if __name__ == "__main__":
    main()