import streamlit as st
import os
import shutil
import tempfile
# import chromadb
# from sentence_transformers import SentenceTransformer
# from src.config import (
#     CHROMA_DB_DIR, EMBEDDING_MODEL, TOP_K_RESULTS,
#     PDFS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
# )
from src.config import (
    EMBEDDING_MODEL, TOP_K_RESULTS,
    PDFS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
)
from src.pdf_processor import load_pdf, chunk_text
# from src.vector_store import search_similar_chunks, get_collection_count
from src.vector_store import (
    search_similar_chunks, get_collection_count,
    get_indexed_papers, delete_paper,
    reset_database, add_chunks_to_db
)
from src.gemini_handler import get_answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Lens",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Dark background */
.stApp {
    background-color: #0f1117;
    color: #e8e6e1;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

/* Cards */
.paper-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #8b949e;
}

.source-card strong {
    color: #58a6ff;
}

.answer-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 1.02rem;
    line-height: 1.8;
    color: #e8e6e1;
}

.badge {
    background: #1f6feb;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
}

.stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e8e6e1 !important;
    border-radius: 8px !important;
}

.stButton > button {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
}

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px;
}

.stExpander {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: ingest a single PDF ───────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def ingest_pdf(filepath, filename):
    """Process one PDF and add to Pinecone"""
    # Check if already ingested
    existing = get_indexed_papers()
    if filename in existing:
        return 0

    pages = load_pdf(filepath)
    chunks = chunk_text(pages)

    if not chunks:
        return 0

    return add_chunks_to_db(chunks, filename)

# def ingest_pdf(filepath, filename):
#     """Process one PDF and add to ChromaDB"""
#     embedder = get_embedder()
#     client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
#     collection = client.get_or_create_collection(name="medical_papers")

#     # Check if already ingested
#     existing = collection.get(where={"source": filename})
#     if existing["ids"]:
#         return 0  # already exists

#     pages = load_pdf(filepath)
#     chunks = chunk_text(pages)

#     if not chunks:
#         return 0

#     texts = [c["text"] for c in chunks]
#     embeddings = embedder.encode(texts)

#     # Use unique IDs based on filename
#     start_id = collection.count()
#     collection.add(
#         ids=[f"{filename}_chunk_{i+start_id}" for i in range(len(chunks))],
#         embeddings=embeddings.tolist(),
#         documents=texts,
#         metadatas=[{
#             "source": filename,
#             "page_number": c["page_number"]
#         } for c in chunks]
#     )
#     return len(chunks)


# def get_indexed_papers():
#     """Get list of unique paper names in DB"""
#     try:
#         client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
#         collection = client.get_or_create_collection(name="medical_papers")
#         results = collection.get()
#         if not results["metadatas"]:
#             return []
#         papers = list(set(m["source"] for m in results["metadatas"]))
#         return sorted(papers)
#     except:
#         return []


# def delete_paper(filename):
#     """Remove a paper's chunks from ChromaDB"""
#     client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
#     collection = client.get_or_create_collection(name="medical_papers")
#     results = collection.get(where={"source": filename})
#     if results["ids"]:
#         collection.delete(ids=results["ids"])


# def reset_database():
#     """Wipe entire ChromaDB"""
#     if os.path.exists(CHROMA_DB_DIR):
#         shutil.rmtree(CHROMA_DB_DIR)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1rem 0;'>
    <h1 style='font-size:2.6rem; margin:0; color:#e8e6e1;'>🔬 Paper Lens</h1>
    <p style='color:#8b949e; font-size:1.05rem; margin-top:6px;'>
        Ask questions across your medical research papers — with citations.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📤 Upload Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = []
        indexed_papers = get_indexed_papers()

        for f in uploaded_files:
            if f.name not in indexed_papers:
                new_files.append(f)

        if new_files:
            if st.button(f"⚡ Ingest {len(new_files)} New Paper(s)", type="primary", use_container_width=True):
                os.makedirs(PDFS_DIR, exist_ok=True)
                progress = st.progress(0)
                total_chunks = 0

                for i, f in enumerate(new_files):
                    with st.spinner(f"Processing {f.name}..."):
                        # Save temp file
                        tmp_path = os.path.join(PDFS_DIR, f.name)
                        with open(tmp_path, "wb") as out:
                            out.write(f.read())

                        chunks_added = ingest_pdf(tmp_path, f.name)
                        total_chunks += chunks_added
                        progress.progress((i + 1) / len(new_files))

                st.success(f"✅ Added {total_chunks} chunks from {len(new_files)} paper(s)!")
                st.rerun()
        else:
            st.info("All uploaded papers already indexed!")

    st.divider()

    # ── Indexed Papers ────────────────────────────────────────────────────────
    st.markdown("### 📚 Indexed Papers")
    papers = get_indexed_papers()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", len(papers))
    with col2:
        st.metric("Chunks", get_collection_count())

    if papers:
        for paper in papers:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"<small style='color:#8b949e'>📄 {paper[:28]}{'...' if len(paper)>28 else ''}</small>", unsafe_allow_html=True)
            with c2:
                if st.button("🗑️", key=f"del_{paper}", help=f"Remove {paper}"):
                    delete_paper(paper)
                    # Also remove from pdfs folder
                    pdf_path = os.path.join(PDFS_DIR, paper)
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    st.rerun()
    else:
        st.markdown("<small style='color:#8b949e'>No papers indexed yet. Upload PDFs above!</small>", unsafe_allow_html=True)

    st.divider()
    if st.button("🔴 Reset All Data", use_container_width=True):
        reset_database()
        if os.path.exists(PDFS_DIR):
            shutil.rmtree(PDFS_DIR)
        st.success("Database cleared!")
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Sources to retrieve", 3, 10, TOP_K_RESULTS)


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
papers = get_indexed_papers()

if not papers:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color:#8b949e;'>
        <div style='font-size:3rem;'>📭</div>
        <h3 style='color:#8b949e; font-family:DM Serif Display,serif;'>No papers indexed yet</h3>
        <p>Upload PDF files using the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g. What are the cardiovascular effects of GLP-1 agonists?",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("🔍 Searching papers..."):
            chunks = search_similar_chunks(query, top_k=top_k)

        with st.spinner("🤖 Generating answer..."):
            answer = get_answer(query, chunks)

        # Answer
        st.markdown("### 💡 Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

        # Sources
        st.markdown("### 📄 Sources Used")
        for i, chunk in enumerate(chunks):
            with st.expander(f"📎 Source {i+1} — {chunk['source']}  |  Page {chunk['page_number']}  |  Score: {chunk['similarity']:.2f}"):
                st.markdown(f"<div class='source-card'>{chunk['text']}</div>", unsafe_allow_html=True)

    elif search_btn and not query:
        st.warning("Please enter a question first!")

    elif not search_btn:
        st.markdown("""
        <div style='text-align:center; padding: 3rem; color:#8b949e;'>
            <div style='font-size:2.5rem;'>🧬</div>
            <p style='margin-top:1rem;'>Type a question above and click <strong>Search</strong></p>
        </div>
        """, unsafe_allow_html=True)