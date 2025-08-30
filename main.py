from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from pydantic import BaseModel
from utils import extract_text_from_pdf, chunk_text, embed_chunks, model
from vectorstore import build_faiss_index, search_index
import requests
import json
import numpy as np
import os
from auth.bearer import verify_bearer_token
import logging
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import time
import asyncio
import aiohttp
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    logger.error("OPENROUTER_API_KEY not found!")
else:
    logger.info(f"OPENROUTER_API_KEY loaded: {api_key[:10]}...")

app = FastAPI(
    title="Document Query System",
    description="High-performance document Q&A system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
router = APIRouter(prefix="/api/v1")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

pdf_cache = {}
cache_lock = threading.Lock()

cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu-")
io_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io-")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

stored_chunks = []
stored_embeddings = []

http_session = None

async def get_http_session():
    global http_session
    if http_session is None:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        http_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
    return http_session

def get_cache_key(content: str) -> str:
    """Generate cache key from content"""
    return hashlib.md5(content.encode()).hexdigest()

def load_from_persistent_cache(cache_key: str):
    """Load from persistent cache"""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None

def save_to_persistent_cache(cache_key: str, data):
    """Save to persistent cache"""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

class QARequest(BaseModel):
    documents: str
    questions: list[str]

async def generate_answer_async(question: str, context: str, retries=2) -> str:
    """Async answer generation with connection reuse"""
    max_context_length = 6000  
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    prompt = f"""Answer the question based on the provided context. If the answer is not in the context, say "Not found in document."

Context: {context}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Document Query System"
    }
    
    data = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {
               "role": "system",
                "content": ( "You are a precise assistant. Answer briefly, but include all necessary factual details from the context. If answer is not found, say exactly: Not found in document."

)

            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 300, 
        "top_p": 0.9
    }

    session = await get_http_session()
    
    for attempt in range(retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data, 
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        
                        if "message" in choice and "content" in choice["message"]:
                            answer = choice["message"]["content"].strip()
                            if answer:
                                return answer
                        
                    return "Unable to generate answer. Please try a different question."
                        
                elif response.status == 429: 
                    if attempt < retries - 1:
                        await asyncio.sleep(5) 
                        continue
                    else:
                        return "Error: Rate limit exceeded. Please try again later."
                        
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP {response.status}: {error_text}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        return f"Error: HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            logger.error(f"Request timeout (attempt {attempt + 1})")
            if attempt < retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                return "Error: Request timeout"
                
        except Exception as e:
            logger.error(f"Request error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: Unable to generate answer after multiple attempts."

async def process_document_fast(doc_url: str):
    """Fast document processing with caching"""
    cache_key = get_cache_key(doc_url)
    
    cached_data = load_from_persistent_cache(cache_key)
    if cached_data:
        logger.info("Using persistent cache")
        return cached_data
    
    with cache_lock:
        if doc_url in pdf_cache:
            logger.info("Using in-memory cache")
            return pdf_cache[doc_url]
    
    logger.info("Processing document...")
    loop = asyncio.get_event_loop()
    
    text = await loop.run_in_executor(io_executor, extract_text_from_pdf, doc_url)
    
    if not text.strip():
        raise ValueError("No text extracted from PDF.")
    
    logger.info(f"Extracted {len(text)} characters.")
    
    chunks_task = loop.run_in_executor(cpu_executor, chunk_text, text)
    chunks = await chunks_task
    
    embeddings = await loop.run_in_executor(cpu_executor, embed_chunks, chunks)
    embeddings_array = np.array(embeddings)
    
    await loop.run_in_executor(cpu_executor, build_faiss_index, embeddings_array)
    
    result = (chunks, embeddings_array)
    
    with cache_lock:
        pdf_cache[doc_url] = result
    
    save_to_persistent_cache(cache_key, result)
    
    logger.info("Document processing completed.")
    return result

async def process_question_fast(question: str, chunks: list, embeddings: np.ndarray):
    """Fast question processing with parallel execution"""
    loop = asyncio.get_event_loop()
    
    query_embedding = await loop.run_in_executor(
        cpu_executor, 
        lambda: model.encode([question])[0]
    )
    
    top_indices = await loop.run_in_executor(
        cpu_executor,
        search_index,
        query_embedding,
        20,
        0.3
    )
    
    if not top_indices:
        return "Not found in document."
    
    candidate_chunks = [chunks[idx] for idx in top_indices]
    
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = await loop.run_in_executor(
        cpu_executor,
        reranker.predict,
        pairs
    )
    
    reranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    
    top_chunks = "\n".join([chunk for _, chunk in reranked[:3]])
    
    answer = await generate_answer_async(question, top_chunks)
    return answer

@router.post("/hackrx/run")
async def ask_questions(
    request: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Optimized endpoint with parallel processing"""
    start_time = time.time()
    
    logger.info("Received /hackrx/run request")
    logger.info(f"Document: {request.documents}")
    logger.info(f"Questions: {len(request.questions)} questions")

    try:
        chunks, embeddings = await process_document_fast(request.documents)
        
        question_tasks = [
            process_question_fast(question, chunks, embeddings)
            for question in request.questions
        ]
        
        answers = await asyncio.gather(*question_tasks, return_exceptions=True)
        
        final_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Error processing question {i+1}: {answer}")
                final_answers.append("Error processing question.")
            else:
                final_answers.append(answer)
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return {"answers": final_answers}

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()
    
    cpu_executor.shutdown(wait=True)
    io_executor.shutdown(wait=True)

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    
    logger.info("Starting Document Query System...")
    logger.info("Warming up models...")
    
    try:
        model.encode(["test query"])
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning(f"Failed to warm up embedding model: {e}")
    
    try:
        reranker.predict([("test question", "test context")])
        logger.info("Reranker model warmed up")
    except Exception as e:
        logger.warning(f"Failed to warm up reranker: {e}")
    
    logger.info("System ready!")