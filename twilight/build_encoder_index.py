"""
Build encoder-based resume index using the trained scratch encoder.
Run after train_encoder.py completes.

Run: /opt/llm-training/bin/python3 build_encoder_index.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from encoder_model import ScratchEncoder, EncoderConfig

ROOT       = Path.home() / "twilight"
PILOT_DIR  = ROOT / "scratch_resume_lm_pilot"
ENC_DIR    = ROOT / "scratch_encoder"
RAG_INDEX  = ROOT / "full_resume_rag_index.npz"
OUT_INDEX  = ROOT / "encoder_resume_index.npz"
DEVICE     = torch.device("cuda:0")
MAX_LEN    = 256
BATCH_SIZE = 64

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(str(PILOT_DIR / "tokenizer.json"))
PAD_ID = tokenizer.token_to_id("<|pad|>") or 0

def encode_batch(texts, max_len=MAX_LEN):
    all_ids, all_masks = [], []
    for text in texts:
        ids = tokenizer.encode(text[:2000]).ids[:max_len]
        pad = max_len - len(ids)
        all_ids.append(ids + [PAD_ID] * pad)
        all_masks.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(all_ids,   dtype=torch.long),
        torch.tensor(all_masks, dtype=torch.long),
    )

# ── Load encoder ───────────────────────────────────────────────────────────────
print("Loading trained encoder...")
ckpt   = torch.load(ENC_DIR / "best.pt", map_location="cpu", weights_only=False)
config = ckpt["config"]
model  = ScratchEncoder(config).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Loaded epoch {ckpt['epoch']}")

# ── Load existing RAG index for resume texts ───────────────────────────────────
print("Loading resume texts from RAG index...")
idx_data = np.load(RAG_INDEX, allow_pickle=True)
rag_meta = json.loads(str(idx_data["metadata"]))
rag_docs = rag_meta["documents"]

# Build full text per resume from chunks
resume_records = []
for doc in rag_docs:
    full_text = " ".join(doc.get("chunks", []))
    resume_records.append({
        "resume_id":      doc["resume_id"],
        "student_folder": doc.get("student_folder", ""),
        "filename":       doc["filename"],
        "full_text":      full_text,
        "chunks":         doc.get("chunks", []),
    })

print(f"  {len(resume_records)} resumes to encode")

# ── Encode all resumes ─────────────────────────────────────────────────────────
all_embeddings = []
texts = [r["full_text"] for r in resume_records]

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding resumes"):
    batch_texts = texts[i:i + BATCH_SIZE]
    ids, masks = encode_batch(batch_texts)
    ids, masks = ids.to(DEVICE), masks.to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embs = model(ids, masks).float().cpu().numpy()
    all_embeddings.append(embs)

all_embeddings = np.vstack(all_embeddings).astype(np.float32)
print(f"  Embedding matrix shape: {all_embeddings.shape}")

# ── Save index ─────────────────────────────────────────────────────────────────
meta_docs = [{k: v for k, v in r.items() if k != "full_text"} for r in resume_records]
metadata = json.dumps({
    "format_version": 1,
    "encoder": "scratch_encoder",
    "embedding_dim": all_embeddings.shape[1],
    "total_resumes": len(resume_records),
    "documents": meta_docs,
})

np.savez(str(OUT_INDEX), metadata=np.array(metadata), embeddings=all_embeddings)
print(f"Saved: {OUT_INDEX}")
print(f"Size: {OUT_INDEX.stat().st_size / 1e6:.1f} MB")
