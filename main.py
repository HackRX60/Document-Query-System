from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from pydantic import BaseModel
from utils import extract_text_from_pdf, chunk_text, embed_chunks, model
from vectorstore import build_faiss_index, search_index
import google.generativeai as genai
import numpy as np
import os
from auth.bearer import verify_bearer_token
import logging
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import time

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found!")
else:
    logger.info(f"GEMINI_API_KEY loaded: {api_key[:10]}...")

genai.configure(api_key=api_key)

# FastAPI app and router
app = FastAPI()
router = APIRouter(prefix="/api/v1")

# In-memory cache
pdf_cache = {}

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Global session storage
stored_chunks = []
stored_embeddings = []

# Request schema
class QARequest(BaseModel):
    documents: str
    questions: list[str]

# Answer generator using Gemini
def generate_answer(question: str, context: str, retries=3) -> str:
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {question}

If the answer is not present in the context, reply: "Not found in document."

Answer clearly and briefly in one sentence, based only on the context.
""".strip()


    for attempt in range(retries):
        try:
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                return "Not found in document."
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            if "quota" in str(e).lower() and attempt < retries - 1:
                sleep_time = 12 * (attempt + 1)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return "Error generating answer."

@router.post("/hackrx/run")
async def ask_questions(
    request: QARequest,
    token: str = Depends(verify_bearer_token)
):
    global stored_chunks, stored_embeddings

    logger.info("Received /hackrx/run request")
    logger.info(f"Document: {request.documents}")
    logger.info(f"Questions: {request.questions}")

    try:
        doc_url = request.documents

        if doc_url in pdf_cache:
            logger.info("Using cached document data.")
            stored_chunks, stored_embeddings = pdf_cache[doc_url]
        else:
            logger.info("Extracting text from PDF URL...")
            text = extract_text_from_pdf(doc_url)

            if not text.strip():
                raise ValueError("No text extracted from PDF.")

            logger.info(f"Extracted {len(text)} characters.")
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)

            stored_chunks = chunks
            stored_embeddings = np.array(embeddings)

            pdf_cache[doc_url] = (stored_chunks, stored_embeddings)
            build_faiss_index(stored_embeddings)
            logger.info("FAISS index built and cached.")

        answers = []

        for i, question in enumerate(request.questions):
            logger.info(f"Processing Q{i+1}: {question}")

            query_embedding = model.encode([question])[0]
            top_indices = search_index(query_embedding, top_k=30, threshold=0.3)
            candidate_chunks = [stored_chunks[idx] for idx in top_indices]

            if not candidate_chunks:
                answers.append({"question": question, "answer": "Not found in document."})
                continue

            # Rerank using cross-encoder
            pairs = [(question, chunk) for chunk in candidate_chunks]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)

            # Log reranked top 5
            logger.info("Top reranked chunks:")
            for score, chunk in reranked[:5]:
                logger.info(f"Score: {score:.2f} | Chunk: {chunk[:100].strip()}")

            top_chunks = "\n".join([chunk for _, chunk in reranked[:5]])
            answer = generate_answer(question, top_chunks)
            answers.append({"question": question, "answer": answer})
            logger.info(f"Answer: {answer}")

        return {"answers": [a["answer"] for a in answers]}

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Register router
app.include_router(router)
