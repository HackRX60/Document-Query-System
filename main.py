from fastapi import FastAPI, Depends, HTTPException, status, Request
from pydantic import BaseModel
from utils import extract_text_from_pdf, chunk_text, embed_chunks, model
from vectorstore import build_faiss_index, search_index
import google.generativeai as genai
import numpy as np
import os
from auth.bearer import verify_bearer_token
import logging

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if API key is loaded
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables!")
else:
    logger.info(f"GEMINI_API_KEY loaded: {api_key[:10]}...")

genai.configure(api_key=api_key)

app = FastAPI()

# Add base URL prefix
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1")

stored_chunks = []
stored_embeddings = []

class QARequest(BaseModel):
    documents: str
    questions: list[str]

@router.post("/hackrx/run")
async def ask_questions(
    request: QARequest,
    token: str = Depends(verify_bearer_token)
    ):
    global stored_chunks, stored_embeddings

    try:
        # Step 1: Extract PDF
        logger.info("Extracting text from PDF...")
        text = extract_text_from_pdf(request.documents)

        # Step 2: Chunk and Embed
        logger.info("Chunking and embedding text...")
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        stored_chunks = chunks
        stored_embeddings = np.array(embeddings)

        # Step 3: Build FAISS index
        logger.info("Building FAISS index...")
        build_faiss_index(stored_embeddings)

        # Step 4: Process each question
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
            query_embedding = model.encode([question])[0]
            top_indices = search_index(query_embedding)
            top_chunks = "\n".join([stored_chunks[i] for i in top_indices])
            
            # Step 5: Gemini one-liner with error handling
            prompt = f"""Context:\n{top_chunks}\n\nQuestion: {question}\nGive a one-line answer:"""
            
            try:
                response = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)
                
                # Check if response has valid content
                if response and hasattr(response, 'text') and response.text:
                    answer = response.text.strip()
                    answers.append({ "question": question, "answer": answer })
                    logger.info(f"Successfully generated answer for question {i+1}")
                else:
                    # Handle empty or invalid response
                    answers.append({ 
                        "question": question, 
                        "answer": "Error: No valid response generated. Please try again." 
                    })
                    logger.warning(f"Empty response for question {i+1}")
                    
            except Exception as e:
                error_msg = f"Gemini API error: {str(e)}"
                logger.error(error_msg)
                answers.append({ 
                    "question": question, 
                    "answer": f"Error: Unable to generate answer. Please try again later. (API Error: {str(e)})" 
                })

        # Extract just the answers from the results
        answer_list = [item["answer"] for item in answers]
        return { "answers": answer_list }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Include the router in the app
app.include_router(router)
