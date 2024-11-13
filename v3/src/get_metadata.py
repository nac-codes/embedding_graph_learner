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

def ocr_pdf_to_text(pdf_path):
    pdf_filename = os.path.basename(pdf_path)
    pdf_dir = os.path.dirname(pdf_path)

    # Define a threshold for average characters per page (e.g., 100)
    avg_chars_per_page_threshold = 200

    text_density = get_text_density(pdf_path)
    if text_density == 0:
        print(f"Unable to process {pdf_filename}. Skipping to OCR.")
        use_ocr = True
    else:
        use_ocr = text_density < avg_chars_per_page_threshold

    if use_ocr:
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

    # Create the output filename based on parent directory
    output_filename = f"{pdf_path.split('/')[-2]}.txt"
    output_text_path = os.path.join(pdf_dir, output_filename)

    # Save the extracted text to the new filename
    with open(output_text_path, 'w') as file_out:
        file_out.write(extracted_text)

    # Save metadata to a .met file
    metadata_filename = output_filename + '.met'
    metadata_path = os.path.join(pdf_dir, metadata_filename)
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    print(f"Extracted metadata and saved text for {pdf_filename} as {output_filename}")

def process_text_file(txt_path):
    txt_filename = os.path.basename(txt_path)
    txt_dir = os.path.dirname(txt_path)

    # Read the text file
    with open(txt_path, 'r') as f:
        text = f.read()

    # Extract metadata using OpenAI on the first part of the text
    metadata = extract_metadata(text, txt_filename)

    # Get user input for unknown metadata
    metadata = get_user_input_for_unknown(metadata)

    # set output filename to the directory of the text file
    output_filename = f"{os.path.split(os.path.dirname(txt_path))[1]}.txt"
    output_text_path = os.path.join(txt_dir, output_filename)

    # Save the text to the new filename
    with open(output_text_path, 'w') as f_out:
        f_out.write(text)

    # Save metadata to a .met file
    metadata_filename = output_filename + '.met'
    metadata_path = os.path.join(txt_dir, metadata_filename)
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    print(f"Extracted metadata and saved text for {txt_filename} as {output_filename}")

def process_corpora(corpora_dir):
    # Loop through the first layer of subdirectories only
    for subdir in os.listdir(corpora_dir):
        subdir_path = os.path.join(corpora_dir, subdir)
        if os.path.isdir(subdir_path):

            # check if metadata and text already exist
            metadata_path = os.path.join(subdir_path, f"{subdir}.txt.met")
            text_path = os.path.join(subdir_path, f"{subdir}.txt")
            if os.path.exists(metadata_path) and os.path.exists(text_path):
                print(f"Skipping {subdir_path} because metadata and text already exist.")
                continue

            # List all PDF and txt files in the subdirectory
            pdf_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.pdf')]
            txt_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.txt')]

            # Process all PDFs
            if pdf_files:
                pdf_file = pdf_files[0]
                pdf_path = os.path.join(subdir_path, pdf_file)
                print(f"Processing PDF: {pdf_path}")
                ocr_pdf_to_text(pdf_path)
            
            # Process only the first .txt file
            elif txt_files:
                txt_file = txt_files[0]
                txt_path = os.path.join(subdir_path, txt_file)
                print(f"Processing Text File: {txt_path}")
                process_text_file(txt_path)



# Main execution
if __name__ == "__main__":
    # Input: corpora directory
    corpora_dir = sys.argv[1] if len(sys.argv) > 1 else input("Enter the directory containing the corpora: ")
    if not os.path.isdir(corpora_dir):
        print("Invalid directory. Exiting.")
    else:
        process_corpora(corpora_dir)
        print("Processing complete for all documents.")
