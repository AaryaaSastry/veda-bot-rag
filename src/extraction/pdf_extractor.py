import pymupdf
import os

def extract_pages_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF file as a list of pages.
    Each page is a string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # We extract text page by page to allow for header/footer detection
        pages.append(page.get_text("text"))
    
    doc.close()
    return pages

