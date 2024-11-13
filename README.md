# Embedding Graph Learner

A tool for processing, chunking, embedding, and interactively learning from text corpora using a graph-based approach with semantic search capabilities.

## Overview

Embedding Graph Learner is a system that:
1. Processes text documents (PDF/TXT)
2. Extracts metadata and chunks content
3. Creates embeddings using OpenAI's API
4. Builds a semantic graph connecting related content
5. Provides an interactive learning interface with AI-assisted explanations

## Features

- **Document Processing**: Handles both PDF and TXT files with OCR capabilities
- **Metadata Extraction**: Automatically extracts author, title, and publication info with GPT-4
- **Smart Chunking**: Splits documents into semantic chunks with configurable overlap
- **Semantic Graph**: Creates a graph where nodes are text chunks and edges represent semantic similarity
- **Interactive Learning**: 
  - Search through content using natural language
  - Get AI-powered explanations of content
  - Take personal notes
  - Track learning progress
  - Literature review mode for multiple related passages

## Installation

```bash
git clone https://github.com/yourusername/embedding_graph_learner.git
cd v3/src
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Documents

Create a directory structure:
```
corpora/
  ├── document1/
  │   └── document1.pdf
  ├── document2/
  │   └── document2.txt
  └── ...
```

### 2. Process Documents & Extract Metadata

```bash
python src/get_metadata.py --corpora-dir ./corpora
```

This will:
- Extract text from PDFs (using OCR if needed)
- Generate metadata files (.met)
- Create standardized text files

### 3. Create Chunks & Embeddings

```bash
python src/get_chunks.py --corpora-dir ./corpora --chunk-size 2000 --overlap-ratio 0.1
```

This creates for each document:
- `{document}_chunks.json`: Text chunks with metadata
- `{document}_embeddings.npy`: Corresponding embeddings

### 4. Build the Graph

```bash
python src/make_graph.py --corpora-dir ./corpora --output corpora_graph.gpickle --n-neighbors 5
```

Creates a graph where:
- Nodes contain chunks, metadata, and embeddings
- Edges connect semantically similar content

### 5. Create Chunk File Paths

```bash
python src/create_chunk_paths.py --corpora-dir ./corpora --graph-file corpora_graph.gpickle
```

### 6. Interactive Learning

```bash
python src/read.py corpora_graph.gpickle
```

In the interactive mode, you can:
- Press Enter to explore random content
- Type a topic/question to search relevant content
- For each chunk:
  - Press [E] for AI explanation
  - Press [N] to take notes
  - Press [S] to skip
  - Press [Q] to quit

The system will save your progress and notes automatically.

## Key Components

The system consists of several main scripts:

1. `get_metadata.py`: Document processing and metadata extraction

```1:82:v3/src/get_metadata.py
import os
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import json
from openai import OpenAI
import re
import sys

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

def get_text_density(pdf_path):
    """
    Calculate the average number of characters per page in the PDF.
    """
    char_count = 0
    page_count = 0
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        char_count += len(text.strip())
                        page_count += 1
                except Exception as e:
                    print(f"Error extracting text from page in {pdf_path}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return 0  # Return 0 to indicate an error occurred
    return char_count / page_count if page_count > 0 else 0

def extract_metadata(text, filename):
    prompt = f"""
Based on the following text extracted from a document and its filename, please extract the author name, title, publication date, and publisher. If you can't find a specific piece of information, use 'Unknown' for that field.

Filename: {filename}

Extracted text:
{text[:2000]}  # Using the first 2000 characters

Please respond in JSON format with the following structure:
{{
    "author": "Author name",
    "title": "Document title",
    "publication_date": "YYYY-MM-DD or Unknown",
    "publisher": "Publisher name or Unknown"
}}
"""
    print(f"Prompt: {prompt}")
    response = query_openai(prompt)
    # Strip response of ```json and ``` if they are there
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
            "publication_date": "Unknown",
            "publisher": "Unknown"
        }
```


2. `get_chunks.py`: Text chunking and embedding generation

```96:131:v3/src/get_chunks.py
    main()
```


3. `make_graph.py`: Graph construction and semantic connection

```63:115:v3/src/make_graph.py
    return all_chunks, np.array(all_embeddings)
```


4. `read.py`: Interactive learning interface

```373:472:v3/src/read.py
        return top_similarities[0][0]
```


## Notes

- Requires OpenAI API key set in environment
- Uses GPT-4o for explanations and GPT-4o-mini for search
- Progress and notes are automatically saved
- Can restore progress from previous sessions

## License

MIT