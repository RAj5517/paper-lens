from src.pdf_processor import load_all_pdfs
from src.vector_store import add_chunks_to_db, get_collection_count

print("=== Medical RAG - Ingestion Pipeline ===\n")

# Check if already ingested
count = get_collection_count()
if count > 0:
    print(f"⚠️  Database already has {count} chunks!")
    answer = input("Re-ingest? This will reset the database. (y/n): ")
    if answer.lower() != 'y':
        print("Skipping ingestion.")
        exit()

# Load and chunk PDFs
chunks = load_all_pdfs()

if chunks:
    # Store in vector DB
    add_chunks_to_db(chunks)
    print(f"\n✅ Ingestion complete! {get_collection_count()} chunks ready.")
else:
    print("❌ No chunks found. Add PDFs to pdfs/ folder first!")