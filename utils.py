import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List
import numpy as np
from functools import lru_cache

# Download NLTK data once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 256 

executor = ThreadPoolExecutor(max_workers=4)

@lru_cache(maxsize=128)
def cached_tokenize(text_hash: str, text: str):
    """Cache sentence tokenization results"""
    return sent_tokenize(text)

async def extract_text_from_pdf_async(url: str):
    """Async PDF download and extraction"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                pdf_content = await response.read()
                
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    executor, 
                    _extract_text_from_bytes, 
                    pdf_content
                )
                return text
                
        except Exception as e:
            raise ValueError(f"Failed to download/extract PDF: {e}")

def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes - runs in thread pool"""
    try:
        pdf_file = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in pdf_file]
        pdf_file.close()
        return "\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

def extract_text_from_pdf(url: str):
    """Synchronous wrapper for backward compatibility"""
    import requests
    from io import BytesIO

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        pdf_content = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            pdf_content.write(chunk)
        
        pdf_content.seek(0)
        return _extract_text_from_bytes(pdf_content.getvalue())
        
    except Exception as e:
        raise ValueError(f"Failed to download/extract PDF: {e}")

def chunk_text_fast(text: str, max_tokens=100, overlap=20):
    """Optimized chunking with better performance"""
    text_hash = str(hash(text))
    sentences = cached_tokenize(text_hash, text)
    
    chunks = []
    chunk = []
    current_tokens = 0
    
    sentence_tokens = [len(sentence.split()) for sentence in sentences]
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        tokens = sentence_tokens[i]
        
        if current_tokens + tokens > max_tokens and chunk:
            chunks.append(" ".join(chunk))
            
            if len(chunk) > overlap:
                chunk = chunk[-overlap:]
                current_tokens = sum(sentence_tokens[j] for j in range(i-overlap, i))
            else:
                chunk = []
                current_tokens = 0
        
        chunk.append(sentence)
        current_tokens += tokens
        i += 1
    
    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks

def embed_chunks_batch(chunks: List[str], batch_size: int = 32):
    """Batch embedding for better performance"""
    if not chunks:
        return []
    
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(
            batch, 
            convert_to_tensor=False,
            show_progress_bar=False, 
            batch_size=batch_size
        )
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def embed_chunks(chunks):
    """Backward compatible embedding function"""
    return embed_chunks_batch(chunks)


def process_multiple_pdfs_parallel(urls: List[str], max_workers: int = None):
    """Process multiple PDFs in parallel"""
    if max_workers is None:
        max_workers = min(len(urls), mp.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_text_from_pdf, urls))
    
    return results

def chunk_text_streaming(text: str, max_tokens=100, overlap=20):
    """Memory-efficient chunking for very large texts"""
    import io
    
    text_stream = io.StringIO(text)
    sentences = []
    chunk = []
    current_tokens = 0
    
    
    for line in text_stream:
        line_sentences = sent_tokenize(line.strip())
        for sentence in line_sentences:
            tokens = len(sentence.split())
            
            if current_tokens + tokens > max_tokens and chunk:
                yield " ".join(chunk)
                
                
                if len(chunk) > overlap:
                    chunk = chunk[-overlap:]
                    current_tokens = sum(len(s.split()) for s in chunk)
                else:
                    chunk = []
                    current_tokens = 0
            
            chunk.append(sentence)
            current_tokens += tokens
    
    if chunk:
        yield " ".join(chunk)

chunk_text = chunk_text_fast 
