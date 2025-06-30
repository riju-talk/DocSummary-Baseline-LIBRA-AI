import re
from PyPDF2 import PdfReader

def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            # Clean extra spaces and line breaks
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            text += page_text + " "
        return text.strip()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks