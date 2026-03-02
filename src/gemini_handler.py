# from google import genai
# from src.config import GEMINI_API_KEY, GEMINI_MODEL

# client = genai.Client(api_key=GEMINI_API_KEY)

# def build_prompt(query, retrieved_chunks):
#     context = ""
#     for i, chunk in enumerate(retrieved_chunks):
#         context += f"""
# Source {i+1}: {chunk['source']} (Page {chunk['page_number']})
# {chunk['text']}
# ---"""

#     prompt = f"""You are a medical research assistant. Answer the question using ONLY the provided context from research papers.

# RULES:
# - Answer ONLY from the context below
# - Always cite sources like [Source 1, Page X]
# - If answer is not in context, say "I cannot find this in the provided papers"
# - Never use your own knowledge, only the papers

# CONTEXT:
# {context}

# QUESTION: {query}

# ANSWER:"""
#     return prompt

# def get_answer(query, retrieved_chunks):
#     prompt = build_prompt(query, retrieved_chunks)
#     response = client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=prompt
#     )
#     return response.text

from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(query, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
Source {i+1}: {chunk['source']} (Page {chunk['page_number']})
{chunk['text']}
---"""
    return f"""You are a medical research assistant. Answer using ONLY the context below.
RULES:
- Cite sources like [Source 1, Page X]
- If not in context say "I cannot find this in the provided papers"

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""

def get_answer(query, retrieved_chunks):
    prompt = build_prompt(query, retrieved_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # free & powerful
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content