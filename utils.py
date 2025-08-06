import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(url: str):
    import requests
    from io import BytesIO
    response = requests.get(url)
    pdf_file = fitz.open(stream=BytesIO(response.content), filetype="pdf")
    return "\n".join([page.get_text() for page in pdf_file])

def chunk_text(text: str, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks, chunk, count = [], [], 0
    for sentence in sentences:
        count += len(sentence.split())
        chunk.append(sentence)
        if count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=False)
