import faiss
import numpy as np
import os

INDEX_PATH = "faiss_index/index_file"

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    return index

def load_faiss_index():
    return faiss.read_index(INDEX_PATH)

def search_index(query_embedding, top_k=3):
    index = load_faiss_index()
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0]
