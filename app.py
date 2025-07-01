from flask import Flask, request, jsonify
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from utils.chunker import extract_text, chunk_text
from retriever.indexer import VectorIndexer
from models.model_loader import QAModel

# Initialize components
os.makedirs("data/docs", exist_ok=True)
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_index.bin"

# Initialize files
if not os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "w") as f:
        json.dump({"chunks": [], "metadata": {}}, f)

indexer = VectorIndexer()
qa_model = QAModel()

app = Flask(__name__)

@app.post("/upload")
def upload_file():
    """Handle file upload and indexing"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        # Save file with unique name
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        save_path = os.path.join("data/docs", f"{file_id}{file_ext}")
        file.save(save_path)
        
        # Process file
        text = extract_text(save_path)
        chunks = chunk_text(text)
        
        # Update chunks database
        with open(CHUNKS_PATH, "r") as f:
            data = json.load(f)
            
        metadata = {
            "file_id": file_id,
            "original_name": file.filename,
            "upload_time": datetime.now().isoformat(),
            "chunk_count": len(chunks)
        }
        
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "id": f"{file_id}_{i}",
                "text": chunk,
                "metadata": metadata
            })
            
        data["chunks"].extend(chunk_records)
        data["metadata"][file_id] = metadata
        
        with open(CHUNKS_PATH, "w") as f:
            json.dump(data, f, indent=2)
            
        # Rebuild index with all chunks
        all_chunks = [c["text"] for c in data["chunks"]]
        indexer.build_index(all_chunks, INDEX_PATH)
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "chunk_count": len(chunks)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/summarize")
def summarize_document():
    """Summarize the first few chunks of the most recent document"""
    try:
        with open(CHUNKS_PATH, "r") as f:
            data = json.load(f)
            
        if not data["chunks"]:
            return jsonify({"error": "No documents available"}), 404
            
        # Get chunks from most recent document
        latest_file = max(data["metadata"].items(), key=lambda x: x[1]["upload_time"])
        file_chunks = [c["text"] for c in data["chunks"] if c["metadata"]["file_id"] == latest_file[0]]
        
        # Summarize first 3 chunks (to stay within CPU limits)
        summary = qa_model.summarize("\n\n".join(file_chunks[:3]))
        return jsonify({"summary": summary})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/qa")
def answer_question():
    """Answer a question based on document content"""
    question = request.args.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    try:
        with open(CHUNKS_PATH, "r") as f:
            data = json.load(f)
            
        if not data["chunks"]:
            return jsonify({"error": "No documents available"}), 404
            
        # Retrieve relevant chunks
        chunk_indices = indexer.retrieve(question, k=3, index_file=INDEX_PATH)
        context_chunks = [data["chunks"][i]["text"] for i in chunk_indices if i < len(data["chunks"])]
        
        if not context_chunks:
            return jsonify({"error": "No relevant context found"}), 404
            
        # Generate answer
        answer = qa_model.answer_question("\n\n".join(context_chunks), question)
        return jsonify({
            "question": question,
            "answer": answer,
            "source_chunks": context_chunks
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)