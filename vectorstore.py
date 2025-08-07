import faiss
import numpy as np
import os

INDEX_PATH = "faiss_index/index_file"

# Global variable to avoid reloading
index = None

def build_faiss_index(embeddings: np.ndarray):
    global index
    dimension = embeddings.shape[1]

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity if normalized
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    return index

def load_faiss_index():
    global index
    if index is None:
        index = faiss.read_index(INDEX_PATH)
    return index

def search_index(query_embedding, top_k=5, threshold=0.6):
    index = load_faiss_index()

    # Normalize query vector
    query = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query)

    distances, indices = index.search(query, top_k)

    # Filter based on similarity threshold
    filtered_indices = [
        idx for idx, score in zip(indices[0], distances[0])
        if score >= threshold
    ]

    # Optional: print matching scores for debug
    print("Top Matches with Scores:")
    for idx, score in zip(indices[0], distances[0]):
        print(f"Chunk #{idx} -> Score: {score}")

    return filtered_indices
