import fitz  # PyMuPDF
import os
from src.config import PDFS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(filepath):
    """Extract text from PDF page by page"""
    doc = fitz.open(filepath)
    pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip empty pages
            pages.append({
                "text": text,
                "page_number": page_num + 1,
                "source": os.path.basename(filepath)
            })
    
    doc.close()
    return pages

def chunk_text(pages):
    """Split pages into overlapping chunks"""
    chunks = []
    
    for page in pages:
        words = page["text"].split()
        
        # slide window over words
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 100:  # skip tiny chunks
                chunks.append({
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "source": page["source"]
                })
            
            start += CHUNK_SIZE - CHUNK_OVERLAP  # overlap
    
    return chunks

def load_all_pdfs():
    """Load and chunk all PDFs from pdfs/ folder"""
    all_chunks = []
    pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in pdfs/ folder!")
        return []
    
    for pdf_file in pdf_files:
        filepath = os.path.join(PDFS_DIR, pdf_file)
        print(f"Processing: {pdf_file}")
        
        pages = load_pdf(filepath)
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)
        
        print(f"  → {len(pages)} pages, {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks