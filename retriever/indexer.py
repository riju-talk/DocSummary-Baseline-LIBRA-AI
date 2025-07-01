import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List

class VectorIndexer:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def build_index(self, chunks: List[str], index_file: str = "data/faiss_index.bin"):
        """Build and save FAISS index"""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
            
        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Using Inner Product for similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)
        
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        faiss.write_index(index, index_file)
        return len(chunks)
        
    def retrieve(
        self, 
        query: str, 
        k: int = 3, 
        index_file: str = "data/faiss_index.bin"
    ) -> List[str]:
        """Retrieve top-k most relevant chunks"""
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        index = faiss.read_index(index_file)
        query_embed = self.embed_model.encode([query])
        faiss.normalize_L2(query_embed)
        
        distances, indices = index.search(query_embed, k)
        return indices[0].tolist()