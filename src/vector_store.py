import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, TOP_K_RESULTS

# Load embedding model once (downloads first time ~80MB)
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready!")

# Setup ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="medical_papers")

def add_chunks_to_db(chunks):
    """Embed all chunks and store in ChromaDB"""
    if not chunks:
        print("No chunks to add!")
        return

    print(f"Embedding {len(chunks)} chunks... (this takes a few minutes)")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Add to ChromaDB
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "source": chunk["source"],
            "page_number": chunk["page_number"]
        } for chunk in chunks]
    )

    print(f"✅ {len(chunks)} chunks stored in ChromaDB!")

def search_similar_chunks(query, top_k=TOP_K_RESULTS):
    """Find most relevant chunks for a query"""

    # Embed the query
    query_embedding = embedder.encode([query])[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Format results
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "similarity": 1 - results["distances"][0][i]
        })

    return chunks

def get_collection_count():
    """How many chunks are stored"""
    return collection.count()