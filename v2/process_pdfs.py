import os
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import json
from openai import OpenAI
import re

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Initialize OpenAI client
client = OpenAI()

def query_openai(prompt, model_name="gpt-4o", assistant_instructions="You are a helpful assistant."):
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
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                char_count += len(text.strip())
                page_count += 1
    return char_count / page_count if page_count > 0 else 0

def extract_text_from_pdf(pdf_path, output_text_path, pdf_filename):
    """
    Extract text from a PDF and save it to a text file.
    """
    with open(pdf_path, 'rb') as file_in, open(output_text_path, 'w') as file_out:
        reader = PyPDF2.PdfReader(file_in)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                file_out.write(f"File: --- {pdf_filename} --- Page {page_num + 1} ---\n{text.strip()}\n\n")

def extract_metadata(text, filename):
    prompt = f"""
    Based on the following text extracted from a PDF and its filename, please extract the author name, title, and publication date. If you can't find a specific piece of information, use 'Unknown' for that field.

    Filename: {filename}

    Extracted text:
    {text[:2000]}  # Using the first 1000 characters for brevity

    Please respond in JSON format with the following structure:
    {{
        "author": "Author name",
        "title": "Document title",
        "publication_date": "YYYY-MM-DD or Unknown"
    }}
    """
    
    response = query_openai(prompt)
    # strip response of ```json and ``` if they are there
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
    # Remove or replace characters that are not allowed in filenames
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def ocr_pdf_to_text(pdf_filename):
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    
    # Define a threshold for average characters per page (e.g., 100)
    avg_chars_per_page_threshold = 200
    
    if get_text_density(pdf_path) < avg_chars_per_page_threshold:
        pages = convert_from_path(pdf_path, 300)
        extracted_text = ""
        for page_num, page_img in enumerate(tqdm(pages, desc="Performing OCR", unit="page")):
            text = pytesseract.image_to_string(page_img)
            extracted_text += f"File: --- {pdf_filename} --- Page {page_num + 1} ---\n{text.strip()}\n\n"
        print(f"Performed OCR and processed {pdf_filename}")
    else:
        extracted_text = ""
        with open(pdf_path, 'rb') as file_in:
            reader = PyPDF2.PdfReader(file_in)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted_text += f"File: --- {pdf_filename} --- Page {page_num + 1} ---\n{text.strip()}\n\n"
        print(f"Extracted text from {pdf_filename}")
    
    # Extract metadata using OpenAI
    metadata = extract_metadata(extracted_text, pdf_filename)
    
    # Get user input for unknown metadata
    metadata = get_user_input_for_unknown(metadata)
    
    # Create the output filename based on metadata
    output_filename = f"{sanitize_filename(metadata['author'])}_{sanitize_filename(metadata['title'])}_{sanitize_filename(metadata['publication_date'])}.txt"
    output_text_path = os.path.join(pdf_dir, output_filename)
    
    # Save the extracted text to the new filename
    with open(output_text_path, 'w') as file_out:
        file_out.write(extracted_text)
    
    # Save metadata to a JSON file
    metadata_filename = f"{os.path.splitext(output_filename)[0]}_metadata.json"
    metadata_path = os.path.join(pdf_dir, metadata_filename)
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    
    print(f"Extracted metadata and saved text for {pdf_filename} as {output_filename}")

# Directory containing PDFs
pdf_dir = os.getcwd()  # Current directory

# Process each PDF file in the directory
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith('.pdf'):
        print(f"Processing {pdf_file}...")
        ocr_pdf_to_text(pdf_file)

print("Processing complete for all PDFs.")