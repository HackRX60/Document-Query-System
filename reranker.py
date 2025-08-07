from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the Cross-Encoder reranker model
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name)

def rerank_chunks(query, chunks, top_n=3):
    """
    Reranks a list of chunks for a given query using a Cross-Encoder.
    Returns top_n best-matching chunks.
    """
    # Prepare inputs: [(query, chunk1), (query, chunk2), ...]
    inputs = tokenizer(
        [(query, chunk) for chunk in chunks],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    # Get top N scores and corresponding chunks
    top_indices = torch.topk(scores, top_n).indices.tolist()
    return [chunks[i] for i in top_indices]
