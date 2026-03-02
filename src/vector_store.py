from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from src.config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_MODEL, TOP_K_RESULTS

# Load embedding model once
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready!")

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def add_chunks_to_db(chunks, filename):
    """Embed chunks and upsert to Pinecone"""
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Build vectors for Pinecone
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{filename}_chunk_{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"]
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    return len(vectors)

def search_similar_chunks(query, top_k=TOP_K_RESULTS):
    """Find most relevant chunks for a query"""
    query_embedding = embedder.encode([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "page_number": match["metadata"]["page_number"],
            "similarity": match["score"]
        })

    return chunks

def delete_paper(filename):
    """Delete all chunks for a paper"""
    # Fetch IDs matching filename prefix
    results = index.list(prefix=f"{filename}_chunk_")
    ids = list(results)
    if ids:
        index.delete(ids=ids)

def get_indexed_papers():
    """Get list of unique paper names"""
    try:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] == 0:
            return []

        # Query with a dummy vector to get metadata
        dummy = [0.0] * 384
        results = index.query(
            vector=dummy,
            top_k=10000,
            include_metadata=True
        )
        papers = list(set(
            m["metadata"]["source"]
            for m in results["matches"]
            if "source" in m["metadata"]
        ))
        return sorted(papers)
    except:
        return []

def get_collection_count():
    """Total vectors stored"""
    try:
        stats = index.describe_index_stats()
        return stats["total_vector_count"]
    except:
        return 0

def reset_database():
    """Delete all vectors"""
    index.delete(delete_all=True)




# this is for local use cromadb 

# import chromadb
# from sentence_transformers import SentenceTransformer
# from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, TOP_K_RESULTS

# # Load embedding model once (downloads first time ~80MB)
# print("Loading embedding model...")
# embedder = SentenceTransformer(EMBEDDING_MODEL)
# print("Embedding model ready!")

# # Setup ChromaDB
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_or_create_collection(name="medical_papers")

# def add_chunks_to_db(chunks):
#     """Embed all chunks and store in ChromaDB"""
#     if not chunks:
#         print("No chunks to add!")
#         return

#     print(f"Embedding {len(chunks)} chunks... (this takes a few minutes)")

#     texts = [chunk["text"] for chunk in chunks]
#     embeddings = embedder.encode(texts, show_progress_bar=True)

#     # Add to ChromaDB
#     collection.add(
#         ids=[f"chunk_{i}" for i in range(len(chunks))],
#         embeddings=embeddings.tolist(),
#         documents=texts,
#         metadatas=[{
#             "source": chunk["source"],
#             "page_number": chunk["page_number"]
#         } for chunk in chunks]
#     )

#     print(f"✅ {len(chunks)} chunks stored in ChromaDB!")

# def search_similar_chunks(query, top_k=TOP_K_RESULTS):
#     """Find most relevant chunks for a query"""

#     # Embed the query
#     query_embedding = embedder.encode([query])[0]

#     # Search ChromaDB
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=top_k
#     )

#     # Format results
#     chunks = []
#     for i in range(len(results["documents"][0])):
#         chunks.append({
#             "text": results["documents"][0][i],
#             "source": results["metadatas"][0][i]["source"],
#             "page_number": results["metadatas"][0][i]["page_number"],
#             "similarity": 1 - results["distances"][0][i]
#         })

#     return chunks

# def get_collection_count():
#     """How many chunks are stored"""
#     return collection.count()