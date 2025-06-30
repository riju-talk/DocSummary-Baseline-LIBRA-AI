import os
import json
import uuid
from flask import Flask, request, jsonify, make_response
from utils.chunker import extract_text, chunk_text
from retriever.indexer import build_index, retrieve
from models.mistral_loader import summarize, answer_qa
from dotenv import load_dotenv

load_dotenv()

# Configuration
UPLOAD_FOLDER = "data/docs"
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_index.bin"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize chunks file
if not os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "w") as f:
        json.dump([], f)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1]
    filename = f"{str(uuid.uuid4())}{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    # Process document
    try:
        text = extract_text(save_path)
        chunks = chunk_text(text, size=1024, overlap=128)
        
        with open(CHUNKS_PATH, "r") as f:
            existing_chunks = json.load(f)
        
        updated_chunks = existing_chunks + chunks
        with open(CHUNKS_PATH, "w") as f:
            json.dump(updated_chunks, f)
            
        chunk_count = build_index(CHUNKS_PATH, INDEX_PATH)
        return jsonify({
            "message": "File processed successfully",
            "chunks_added": len(chunks),
            "total_chunks": chunk_count,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['GET'])
def document_summary():
    try:
        with open(CHUNKS_PATH, "r") as f:
            chunks = json.load(f)
        
        if not chunks:
            return jsonify({"error": "No documents processed"}), 400
            
        summary = summarize(chunks)
        # Extract just the summary from model response
        summary = summary.split("[/INST]")[-1].strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/qa', methods=['GET'])
def question_answering():
    question = request.args.get('question', '')
    if not question:
        return jsonify({"error": "Missing question parameter"}), 400
        
    try:
        context_chunks = retrieve(question, k=5, index_file=INDEX_PATH, chunks_file=CHUNKS_PATH)
        context = "\n\n".join(context_chunks)
        answer = answer_qa(question, context)
        # Extract just the answer from model response
        answer = answer.split("Answer:")[-1].strip()
        return jsonify({
            "question": question,
            "answer": answer,
            "context_sources": context_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_system():
    try:
        # Reset chunks
        with open(CHUNKS_PATH, "w") as f:
            json.dump([], f)
        
        # Delete index
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
            
        # Delete documents
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
            
        return jsonify({"message": "System reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    # Use this for production:
    # uvicorn.run("app:app", host="0.0.0.0", port=port, workers=2)
    
    # For development:
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', port, app, use_reloader=True, use_debugger=True)