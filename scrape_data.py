from langchain_community.document_loaders import PyPDFLoader
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PDF_PATH = "data/pune_history.pdf"
OUTPUT_PATH = "data/pune_data.txt"

def extract_text_from_pdf(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        logging.error(f"Error: The file '{pdf_path}' was not found in the project folder.")
        return

    logging.info(f"Loading data from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    logging.info(f"Successfully loaded {len(pages)} pages from the PDF.")
    full_text = "\n".join([page.page_content for page in pages])

    with open("data/pune_data.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    logging.info(f"All text has been extracted and saved to {output_path}")

if __name__ == "__main__":
    extract_text_from_pdf(PDF_PATH, OUTPUT_PATH)