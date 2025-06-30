import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks_file="data/chunks.json", index_file="data/faiss_index.bin"):
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    
    if not chunks:
        return 0
        
    embeds = EMBED_MODEL.encode(chunks, show_progress_bar=True, batch_size=128)
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(np.array(embeds).astype('float32'))
    faiss.write_index(index, index_file)
    return len(chunks)

def retrieve(query: str, k=5, index_file="data/faiss_index.bin", chunks_file="data/chunks.json"):
    if not os.path.exists(index_file):
        return []
        
    index = faiss.read_index(index_file)
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    
    query_embed = EMBED_MODEL.encode([query])
    distances, indices = index.search(np.array(query_embed).astype('float32'), k)
    
    return [chunks[i] for i in indices[0] if i < len(chunks)]