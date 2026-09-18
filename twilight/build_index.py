"""
Build full RAG index from all student resumes.
Run: /opt/llm-training/bin/python3 build_index.py
"""
import json, hashlib, os
from pathlib import Path
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RESUME_DIR = Path.home() / "twilight/Downloaded_Resumes"
OUT_INDEX  = Path.home() / "twilight/full_resume_rag_index.npz"
CHUNK_SIZE = 600   # characters per chunk
CHUNK_OVERLAP = 50

def extract_text(pdf_path):
    try:
        doc = pymupdf.open(str(pdf_path))
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        print(f"  SKIP {pdf_path.name}: {e}")
        return ""

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # Split on sentence boundaries first
    import re
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= size:
            current += (" " if current else "") + sent
        else:
            if len(current) > 60:
                chunks.append(current.strip())
            current = sent
    if len(current) > 60:
        chunks.append(current.strip())
    return chunks if chunks else [text[:size].strip()]

print("Loading sentence encoder...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pdf_files = sorted(RESUME_DIR.rglob("*.pdf"))
print(f"Found {len(pdf_files)} PDF resumes")

documents = []
all_embeddings = []

for pdf_path in tqdm(pdf_files, desc="Indexing resumes"):
    text = extract_text(pdf_path)
    if not text:
        continue
    chunks = chunk_text(text)
    if not chunks:
        continue
    embs = encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    resume_id = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:16]
    student_folder = pdf_path.parent.name  # e.g. Student_0001
    array_key = f"emb_{len(documents)}"
    documents.append({
        "resume_id":     resume_id,
        "student_folder": student_folder,
        "filename":      pdf_path.name,
        "full_text":     text,
        "chunks":        chunks,
        "array_key":     array_key,
    })
    all_embeddings.append((array_key, embs))

print(f"\nIndexed {len(documents)} resumes")
total_chunks = sum(len(d["chunks"]) for d in documents)
print(f"Total chunks: {total_chunks}")

# Save — store embeddings as separate arrays (same format as original index)
meta_docs = [{k: v for k, v in d.items() if k != "full_text"} for d in documents]
metadata = json.dumps({
    "format_version": 2,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "total_resumes": len(documents),
    "total_chunks": total_chunks,
    "documents": meta_docs,
})

save_dict = {"metadata": np.array(metadata)}
for key, embs in all_embeddings:
    save_dict[key] = embs.astype(np.float32)

np.savez(str(OUT_INDEX), **save_dict)
print(f"Saved index: {OUT_INDEX}")
print(f"Index size: {OUT_INDEX.stat().st_size / 1e6:.1f} MB")
