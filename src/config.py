import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
PDFS_DIR = "pdfs"
CHROMA_DB_DIR = "chroma_db"

# Chunking settings
CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # overlap between chunks

# Retrieval settings
TOP_K_RESULTS = 5       # how many chunks to retrieve

# Embedding model (free, runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini model
GEMINI_MODEL = "gemini-2.0-flash"