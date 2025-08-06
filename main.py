from fastapi import FastAPI
from pydantic import BaseModel
from utils import extract_text_from_pdf, chunk_text, embed_chunks, model
from vectorstore import build_faiss_index, search_index
import google.generativeai as genai
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

stored_chunks = []
stored_embeddings = []

class QARequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/ask")
async def ask_questions(request: QARequest):
    global stored_chunks, stored_embeddings

    # Step 1: Extract PDF
    text = extract_text_from_pdf(request.documents)

    # Step 2: Chunk and Embed
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    stored_chunks = chunks
    stored_embeddings = np.array(embeddings)

    # Step 3: Build FAISS index
    build_faiss_index(stored_embeddings)

    # Step 4: Process each question
    answers = []
    for question in request.questions:
        query_embedding = model.encode([question])[0]
        top_indices = search_index(query_embedding)
        top_chunks = "\n".join([stored_chunks[i] for i in top_indices])
        
        # Step 5: Gemini one-liner
        prompt = f"""Context:\n{top_chunks}\n\nQuestion: {question}\nGive a one-line answer:"""
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        answers.append({ "question": question, "answer": response.text.strip() })

    return { "results": answers }
