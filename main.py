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
    # Truncate context if too long (Gemini has input limits)
    max_context_length = 8000  # Conservative limit
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    prompt = f"""Answer the question based on the provided context. If the answer is not in the context, say "Not found in document."

Context: {context}

Question: {question}

Answer:"""

    for attempt in range(retries):
        try:
            # Use generation config for better control
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500,
            }
            
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=generation_config
            )
            
            response = model.generate_content(prompt)
            
            # Better error handling for the specific error you're seeing
            if response and response.candidates:
                candidate = response.candidates[0]
                
                # Check finish reason
                if candidate.finish_reason == 1:  # STOP (normal completion)
                    if candidate.content and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        if text and text.strip():
                            return text.strip()
                        else:
                            logger.warning("Response parts exist but text is empty")
                            return "Unable to generate answer. Please try a different question."
                    else:
                        logger.warning("Response has no content parts")
                        return "Unable to generate answer. Please try a different question."
                        
                elif candidate.finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Response was truncated due to max tokens")
                    return "Answer was too long. Please ask a more specific question."
                    
                elif candidate.finish_reason == 3:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    return "Unable to answer due to content policies."
                    
                elif candidate.finish_reason == 4:  # RECITATION
                    logger.warning("Response blocked due to recitation")
                    return "Unable to answer due to content policies."
                    
                else:
                    logger.warning(f"Unexpected finish reason: {candidate.finish_reason}")
                    return "Unable to generate answer. Please try again later."
            else:
                logger.warning("No response candidates received")
                return "Unable to generate answer. Please try again later."
                
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
            
            # Handle specific API errors
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    sleep_time = 12 * (attempt + 1)
                    logger.info(f"Rate limit hit, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                else:
                    return "Error: Unable to generate answer. Please try again later. (Rate Limit)"
                    
            elif "safety" in str(e).lower() or "blocked" in str(e).lower():
                return "Unable to answer due to content policies."
                
            elif "invalid operation" in str(e).lower():
                return f"Error: Unable to generate answer. Please try again later. (API Error: {str(e)})"
                
            else:
                if attempt < retries - 1:
                    time.sleep(2)  # Brief delay before retry
                    continue
                else:
                    return f"Error: Unable to generate answer. Please try again later. (Error: {str(e)})"
    
    return "Error: Unable to generate answer after multiple attempts."

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
