from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from src.config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_MODEL, TOP_K_RESULTS

_embedder = None
_index = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready!")
    return _embedder

def get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX)
    return _index

def get_namespace(user_id: str) -> str:
    """Convert user email/id to safe Pinecone namespace"""
    return user_id.replace("@", "_at_").replace(".", "_").lower()[:50]

def add_chunks_to_db(chunks, filename, user_id):
    if not chunks:
        return 0
    embedder = get_embedder()
    index = get_index()
    namespace = get_namespace(user_id)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

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

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size], namespace=namespace)

    return len(vectors)

def search_similar_chunks(query, user_id, top_k=TOP_K_RESULTS):
    embedder = get_embedder()
    index = get_index()
    namespace = get_namespace(user_id)

    query_embedding = embedder.encode([query])[0]
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
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

def delete_paper(filename, user_id):
    index = get_index()
    namespace = get_namespace(user_id)
    results = index.list(prefix=f"{filename}_chunk_", namespace=namespace)
    ids = list(results)
    if ids:
        index.delete(ids=ids, namespace=namespace)

def get_indexed_papers(user_id):
    try:
        index = get_index()
        namespace = get_namespace(user_id)
        stats = index.describe_index_stats()

        if namespace not in stats.get("namespaces", {}):
            return []

        dummy = [0.0] * 384
        results = index.query(
            vector=dummy,
            top_k=10000,
            include_metadata=True,
            namespace=namespace
        )
        papers = list(set(
            m["metadata"]["source"]
            for m in results["matches"]
            if "source" in m["metadata"]
        ))
        return sorted(papers)
    except:
        return []

def get_collection_count(user_id):
    try:
        index = get_index()
        namespace = get_namespace(user_id)
        stats = index.describe_index_stats()
        return stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    except:
        return 0

def reset_database(user_id):
    index = get_index()
    namespace = get_namespace(user_id)
    index.delete(delete_all=True, namespace=namespace)