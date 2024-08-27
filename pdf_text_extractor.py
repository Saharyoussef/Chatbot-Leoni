import os
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    # Extracts text from a PDF file at the given path.
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def divide_text_into_sections(text, section_size=500):
    # Divides the extracted text into smaller sections of a specified size.
    sections = [text[i:i + section_size] for i in range(0, len(text), section_size)]
    return sections

def extract_and_divide_pdfs(directory_path, json_output_path, section_size=500):
    # Processes all PDF files in a directory: extracts text, divides it into sections, and saves it as JSON.
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            try:
                text = extract_text_from_pdf(filepath)
                sections = divide_text_into_sections(text, section_size)
                for i, section in enumerate(sections):
                    data.append({
                        "filename": filename,
                        "section_number": i + 1,
                        "section_text": section
                    })
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
    
    # Saves the processed data to a JSON file.
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {json_output_path}")

if __name__ == "__main__":
    # Main function to set paths and call the processing function.
    directory_path = r"CC:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\Reference"
    json_output_path = r"C:\Users\Sahar Y\OneDrive\Bureau\stage leoni\ChatLeoni\file.json"
    extract_and_divide_pdfs(directory_path, json_output_path, section_size=500)
