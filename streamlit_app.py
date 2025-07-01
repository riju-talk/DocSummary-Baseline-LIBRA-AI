# streamlit_app.py
import streamlit as st
import os
import json
import time
from utils.chunker import extract_text, chunk_text
from retriever.indexer import VectorIndexer
from models.model_loader import QAModel

# Constants
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_index.bin"
ALLOWED_EXT = [".pdf", ".txt", ".pptx", ".docx"]

# Ensure folders exist
os.makedirs("data/docs", exist_ok=True)

# Initialize chunks file on first run
if not os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "w") as f:
        json.dump({"chunks": [], "metadata": {}}, f, indent=2)

# Load heavy resources once
@st.cache_resource(show_spinner="Loading Q&A model...")
def load_qa_model():
    return QAModel()

@st.cache_resource(show_spinner="Loading vector indexer...")
def load_indexer():
    return VectorIndexer()

qa_model = load_qa_model()
indexer = load_indexer()

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Streamlit page config
st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
st.title("📄 Document Summarization & QA System")

# Sidebar: file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, DOCX, or PPTX file",
        type=[ext.lstrip('.') for ext in ALLOWED_EXT]
    )

    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in ALLOWED_EXT:
            st.error("Unsupported file type. Please upload PDF, TXT, DOCX, or PPTX.")
        else:
            save_path = os.path.join("data/docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner("Processing document..."):
                    # Extract & chunk
                    text = extract_text(save_path)
                    chunks = chunk_text(text)

                    # Load existing data
                    with open(CHUNKS_PATH, "r") as f:
                        data = json.load(f)

                    # Build metadata
                    metadata = {
                        "file_name": uploaded_file.name,
                        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "chunk_count": len(chunks)
                    }

                    # Create chunk records
                    chunk_records = [
                        {
                            "id": f"{uploaded_file.name}_{i}",
                            "text": chunk,
                            "metadata": metadata
                        }
                        for i, chunk in enumerate(chunks)
                    ]

                    # Update data store
                    data["chunks"].extend(chunk_records)
                    data["metadata"][uploaded_file.name] = metadata
                    with open(CHUNKS_PATH, "w") as f:
                        json.dump(data, f, indent=2)

                    # Rebuild FAISS index
                    all_texts = [c["text"] for c in data["chunks"]]
                    indexer.build_index(all_texts, INDEX_PATH)

                    # Update session state
                    st.session_state.processed = True
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.chunk_count = len(chunks)

                    st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

# Main content
if st.session_state.processed:
    st.header(f"📄 Document: {st.session_state.file_name}")
    st.caption(f"Processed {st.session_state.chunk_count} chunks")

    tab1, tab2 = st.tabs(["🧠 Summarize", "🤖 Ask Questions"])

    # Summarization tab
    with tab1:
        st.subheader("Document Summary")
        with st.spinner("Generating summary..."):
            try:
                with open(CHUNKS_PATH, "r") as f:
                    data = json.load(f)

                # Filter chunks for current file
                file_chunks = [
                    c["text"] for c in data["chunks"]
                    if c["metadata"]["file_name"] == st.session_state.file_name
                ]

                summary = qa_model.summarize("\n\n".join(file_chunks[:3]))
                st.markdown(f"**Summary:**\n\n{summary}")
            except Exception as e:
                st.error(f"Summary error: {e}")

    # QA tab
    with tab2:
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question:", key="question_input")

        if st.button("Get Answer", disabled=not question):
            with st.spinner("Retrieving context and answering..."):
                try:
                    with open(CHUNKS_PATH, "r") as f:
                        data = json.load(f)

                    # Retrieve top-3 chunk indices
                    idxs = indexer.retrieve(
                        question, k=3, index_file=INDEX_PATH
                    )

                    context_chunks = [
                        data["chunks"][i]["text"]
                        for i in idxs if i < len(data["chunks"])
                    ]

                    answer = qa_model.answer_question(
                        "\n\n".join(context_chunks), question
                    )

                    # Save to history
                    st.session_state.qa_history.insert(0, {
                        "question": question,
                        "answer": answer,
                        "timestamp": time.strftime("%H:%M:%S")
                    })

                    st.markdown(f"**Answer:**\n\n{answer}")

                    with st.expander("View Source Context"):
                        for i, txt in enumerate(context_chunks, start=1):
                            st.caption(f"Source {i}:")
                            st.info(txt)
                except Exception as e:
                    st.error(f"Failed to answer: {e}")

        # Display recent Q&A history
        if st.session_state.qa_history:
            st.subheader("Recent Questions")
            for qa in st.session_state.qa_history[:5]:
                st.markdown(
                    f"**Q:** {qa['question']}  \n"
                    f"**A:** {qa['answer']}  \n"
                    f"*⏰ {qa['timestamp']}*"
                )
                st.divider()
else:
    st.info("📂 Upload a document to get started.")
    st.image(
        "https://images.unsplash.com/photo-1507842217343-583bb7270b66"
        "?auto=format&fit=crop&w=1200&h=600",
        caption="Document Analysis System"
    )

# Custom CSS styling
st.markdown("""
<style>
[data-testid="stExpander"] .st-emotion-cache-1qrv4ga {
    background-color: #f0f2f6;
    padding: 12px;
    border-radius: 8px;
}
[data-testid="stFileUploader"] {
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 10px;
}
.stButton button {
    background-color: #4CAF50 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
