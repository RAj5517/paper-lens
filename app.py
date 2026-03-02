import streamlit as st
import os
import re
import shutil
from src.config import (
    EMBEDDING_MODEL, TOP_K_RESULTS,
    PDFS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
)
from src.pdf_processor import load_pdf, chunk_text
from src.vector_store import (
    search_similar_chunks, get_collection_count,
    get_indexed_papers, delete_paper,
    reset_database, add_chunks_to_db
)
from src.gemini_handler import get_answer

st.set_page_config(page_title="Paper Lens", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp { background-color: #070a0f !important; color: #c9d1d9 !important; font-family: 'IBM Plex Mono', monospace !important; }
.stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(0,255,179,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,179,0.03) 1px, transparent 1px); background-size: 40px 40px; pointer-events: none; z-index: 0; }
[data-testid="stSidebar"] { background: #0a0f1a !important; border-right: 1px solid rgba(0,255,179,0.12) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.login-wrap { max-width: 440px; margin: 6rem auto; text-align: center; }
.login-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #f0f6fc; letter-spacing: -0.02em; margin-bottom: 0.5rem; }
.login-title span { color: #00ffb3; }
.login-sub { font-family: 'Instrument Serif', serif; font-style: italic; color: #6e7681; font-size: 1rem; margin-bottom: 2rem; line-height: 1.6; }
.login-card { background: #0d1117; border: 1px solid rgba(0,255,179,0.15); border-radius: 10px; padding: 2rem; }
.user-badge { background: rgba(0,255,179,0.08); border: 1px solid rgba(0,255,179,0.2); border-radius: 6px; padding: 8px 14px; font-size: 0.78rem; color: #00ffb3; display: inline-block; margin-bottom: 1rem; }
.hero { padding: 3.5rem 0 2rem 0; }
.hero-label { font-size: 0.7rem; letter-spacing: 0.25em; text-transform: uppercase; color: #00ffb3; margin-bottom: 1rem; opacity: 0.8; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3.8rem; font-weight: 800; line-height: 1.05; color: #f0f6fc; margin: 0; letter-spacing: -0.02em; }
.hero-title span { color: #00ffb3; }
.hero-sub { font-family: 'Instrument Serif', serif; font-style: italic; font-size: 1.15rem; color: #6e7681; margin-top: 1rem; max-width: 520px; line-height: 1.6; }
.stat-row { display: flex; gap: 12px; margin-top: 2rem; flex-wrap: wrap; }
.stat-pill { background: rgba(0,255,179,0.05); border: 1px solid rgba(0,255,179,0.15); border-radius: 4px; padding: 8px 16px; font-size: 0.75rem; color: #00ffb3; }
.stat-pill strong { color: #f0f6fc; font-weight: 500; }
.stTextInput > div > div > input { background: #0d1117 !important; border: 1px solid rgba(0,255,179,0.2) !important; border-radius: 6px !important; color: #f0f6fc !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.95rem !important; padding: 14px 18px !important; }
.stTextInput > div > div > input:focus { border-color: #00ffb3 !important; box-shadow: 0 0 0 3px rgba(0,255,179,0.08) !important; }
.stTextInput > div > div > input::placeholder { color: #3d444d !important; }
.stButton > button { background: #00ffb3 !important; color: #070a0f !important; border: none !important; border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 500 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; padding: 12px 24px !important; }
.stButton > button:hover { background: #00e6a0 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(0,255,179,0.25) !important; }
.stButton > button[kind="secondary"] { background: transparent !important; color: #6e7681 !important; border: 1px solid #21262d !important; }
.stButton > button[kind="secondary"]:hover { border-color: #ff4d4d !important; color: #ff4d4d !important; transform: none !important; box-shadow: none !important; }
.answer-container { background: #0d1117; border: 1px solid rgba(0,255,179,0.15); border-radius: 8px; padding: 2rem 2.2rem; margin: 1.5rem 0; position: relative; overflow: hidden; }
.answer-container::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: linear-gradient(180deg, #00ffb3, transparent); }
.answer-tag { font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #00ffb3; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px; }
.answer-tag::after { content: ''; flex: 1; height: 1px; background: rgba(0,255,179,0.1); }
.answer-text { font-family: 'Instrument Serif', serif; font-size: 1.08rem; line-height: 1.85; color: #c9d1d9; }
.source-header { font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #6e7681; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px; }
.source-header::after { content: ''; flex: 1; height: 1px; background: #21262d; }
.source-card { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 8px; }
.source-card:hover { border-color: rgba(0,255,179,0.2); }
.source-meta { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.source-file { font-size: 0.75rem; color: #00ffb3; font-weight: 500; }
.source-page { font-size: 0.7rem; color: #6e7681; background: #161b22; padding: 2px 8px; border-radius: 3px; }
.source-text { font-size: 0.82rem; line-height: 1.65; color: #8b949e; border-top: 1px solid #161b22; padding-top: 8px; margin-top: 4px; }
.sidebar-label { font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: #00ffb3; margin-bottom: 10px; opacity: 0.8; }
.paper-item { background: #0d1117; border: 1px solid #21262d; border-radius: 5px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.72rem; color: #8b949e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.custom-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,179,0.15), transparent); margin: 1.5rem 0; }
.empty-state { text-align:center; padding:5rem 2rem; opacity:0.4; }
[data-testid="stFileUploader"] { background: #0d1117 !important; border: 1px dashed rgba(0,255,179,0.2) !important; border-radius: 6px !important; }
.stProgress > div > div { background: #00ffb3 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070a0f; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def ingest_pdf(filepath, filename, user_id):
    existing = get_indexed_papers(user_id)
    if filename in existing:
        return 0
    pages = load_pdf(filepath)
    chunks = chunk_text(pages)
    if not chunks:
        return 0
    return add_chunks_to_db(chunks, filename, user_id)

# ── LOGIN ─────────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    st.markdown("""
    <div class="login-wrap">
        <div style="font-size:3rem;margin-bottom:1.5rem;">🔬</div>
        <div class="login-title">Paper<span>Lens</span></div>
        <div class="login-sub">Your private AI research assistant.<br/>Each user gets their own isolated document space.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<span style="font-size:0.68rem;letter-spacing:0.2em;text-transform:uppercase;color:#6e7681;">// Enter your email to continue</span>', unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed")
        if st.button("→ Enter Paper Lens", use_container_width=True):
            if email and is_valid_email(email):
                st.session_state.user_id = email.lower().strip()
                st.rerun()
            else:
                st.error("Please enter a valid email address.")
        st.markdown('<div style="margin-top:1rem;font-size:0.7rem;color:#3d444d;text-align:center;">Your email isolates your document space.<br/>No password. No data stored.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── APP ───────────────────────────────────────────────────────────────────────
user_id = st.session_state.user_id
user_display = user_id.split("@")[0]

with st.sidebar:
    st.markdown(f'<div class="user-badge">👤 {user_id}</div>', unsafe_allow_html=True)
    if st.button("Sign Out", type="secondary", use_container_width=True):
        st.session_state.user_id = None
        st.rerun()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📤 Upload Papers</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        indexed = get_indexed_papers(user_id)
        new_files = [f for f in uploaded_files if f.name not in indexed]
        if new_files:
            if st.button(f"⚡ Ingest {len(new_files)} Paper(s)", use_container_width=True):
                os.makedirs(PDFS_DIR, exist_ok=True)
                progress = st.progress(0)
                total = 0
                for i, f in enumerate(new_files):
                    tmp = os.path.join(PDFS_DIR, f"{user_id}_{f.name}")
                    with open(tmp, "wb") as out:
                        out.write(f.read())
                    total += ingest_pdf(tmp, f.name, user_id)
                    progress.progress((i + 1) / len(new_files))
                st.success(f"✅ {total} chunks indexed!")
                st.rerun()
        else:
            st.info("All papers already indexed!")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    papers = get_indexed_papers(user_id)
    count = get_collection_count(user_id)
    st.markdown('<div class="sidebar-label">📚 Your Papers</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Papers", len(papers))
    col2.metric("Chunks", count)
    st.markdown("")

    if papers:
        for paper in papers:
            c1, c2 = st.columns([5, 1])
            with c1:
                name = paper[:26] + "…" if len(paper) > 26 else paper
                st.markdown(f'<div class="paper-item">📄 {name}</div>', unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"del_{paper}", help="Remove"):
                    delete_paper(paper, user_id)
                    st.rerun()
    else:
        st.markdown('<p style="font-size:0.75rem;color:#3d444d;">No papers yet.</p>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Sources to retrieve", 3, 10, TOP_K_RESULTS)
    st.markdown("")
    if st.button("🔴 Reset My Data", use_container_width=True, type="secondary"):
        reset_database(user_id)
        st.success("Your data cleared!")
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
papers = get_indexed_papers(user_id)
count = get_collection_count(user_id)

st.markdown(f"""
<div class="hero">
    <div class="hero-label">// Welcome back, {user_display}</div>
    <h1 class="hero-title">Paper<span>Lens</span></h1>
    <p class="hero-sub">Ask anything across your research corpus. Get grounded answers with exact citations — not hallucinations.</p>
    <div class="stat-row">
        <div class="stat-pill"><strong>{len(papers)}</strong> papers indexed</div>
        <div class="stat-pill"><strong>{count}</strong> chunks stored</div>
        <div class="stat-pill">LLM · <strong>Llama 3.3 70B</strong></div>
        <div class="stat-pill">DB · <strong>Pinecone</strong></div>
    </div>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

if not papers:
    st.markdown('<div class="empty-state"><div style="font-size:3rem;">🧬</div><div style="font-family:Syne,sans-serif;font-size:1.2rem;color:#6e7681;">No papers indexed yet</div><div style="font-size:0.8rem;color:#3d444d;margin-top:0.5rem;">Upload PDFs using the sidebar to begin.</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<span style="font-size:0.68rem;letter-spacing:0.2em;text-transform:uppercase;color:#6e7681;">// Ask a question</span>', unsafe_allow_html=True)
    query = st.text_input("Question", placeholder="e.g. What are the cardiovascular effects of GLP-1 agonists?", label_visibility="collapsed")
    search_btn = st.button("→ Search Papers", type="primary")

    if search_btn and query:
        with st.spinner("Searching corpus..."):
            chunks = search_similar_chunks(query, user_id, top_k=top_k)
        with st.spinner("Generating answer..."):
            answer = get_answer(query, chunks)

        st.markdown(f'<div class="answer-container"><div class="answer-tag">AI Answer</div><div class="answer-text">{answer}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="source-header">Retrieved Sources</div>', unsafe_allow_html=True)

        for chunk in chunks:
            score_pct = int(chunk['similarity'] * 100)
            score_color = "#00ffb3" if score_pct > 75 else "#ffa500" if score_pct > 50 else "#6e7681"
            st.markdown(f'<div class="source-card"><div class="source-meta"><span class="source-file">📄 {chunk["source"]}</span><span class="source-page">pg. {chunk["page_number"]}</span><span style="font-size:0.7rem;color:{score_color};margin-left:auto;">{score_pct}% match</span></div><div class="source-text">{chunk["text"][:400]}{"…" if len(chunk["text"]) > 400 else ""}</div></div>', unsafe_allow_html=True)

    elif search_btn and not query:
        st.warning("Enter a question first.")
    else:
        st.markdown('<div style="text-align:center;padding:4rem;opacity:0.3;"><div style="font-size:2rem;">⬆</div><div style="font-size:0.8rem;letter-spacing:0.15em;text-transform:uppercase;margin-top:8px;">Type a question to begin</div></div>', unsafe_allow_html=True)