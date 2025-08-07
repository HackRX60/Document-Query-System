import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(url: str):
    import requests
    from io import BytesIO

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download PDF: {e}")

    try:
        pdf_file = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf_file])
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def chunk_text(text: str, max_tokens=100, overlap=20):
    sentences = sent_tokenize(text)
    chunks, chunk, count = [], [], 0

    for i, sentence in enumerate(sentences):
        chunk.append(sentence)
        count += len(sentence.split())

        if count >= max_tokens:
            chunks.append(" ".join(chunk))
            # Step back by `overlap` sentences
            chunk = sentences[max(0, i - overlap + 1):i+1]
            count = sum(len(s.split()) for s in chunk)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def embed_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=False)
