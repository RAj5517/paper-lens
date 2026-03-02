# import os
# from dotenv import load_dotenv

# load_dotenv()


# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # Paths
# PDFS_DIR = "pdfs"
# CHROMA_DB_DIR = "chroma_db"

# # Chunking settings
# CHUNK_SIZE = 500        # words per chunk
# CHUNK_OVERLAP = 50      # overlap between chunks

# # Retrieval settings
# TOP_K_RESULTS = 5       # how many chunks to retrieve

# # Embedding model (free, runs locally)
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# # Gemini model
# # GEMINI_MODEL = "gemini-2.0-flash"

# # Pinecone
# PINECONE_INDEX = "medical-papers"

# # Groq
# GROQ_MODEL = "llama-3.3-70b-versatile"



import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

PDFS_DIR = "pdfs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PINECONE_INDEX = "medical-papers"
GROQ_MODEL = "llama-3.3-70b-versatile"