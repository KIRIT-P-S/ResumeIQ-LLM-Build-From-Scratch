"""
Resume QA API — scratch SFT v2 + full 440-resume RAG
Run: /opt/llm-training/bin/uvicorn app:app --host 0.0.0.0 --port 8002
"""
import json, re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

ROOT      = Path.home() / "twilight"
PILOT_DIR = ROOT / "scratch_resume_lm_pilot"
SFT_DIR   = ROOT / "scratch_resume_lm_resume_sft_v2"
RAG_INDEX = ROOT / "full_resume_rag_index.npz"
ENC_INDEX = ROOT / "encoder_resume_index.npz"
ENC_DIR   = ROOT / "scratch_encoder"
DEVICE    = torch.device("cuda:0")   # remapped via CUDA_VISIBLE_DEVICES

MAX_NEW_TOKENS       = 60
MAX_NEW_TOKENS_MULTI = 30
REPETITION_PENALTY   = 1.5

# ── Model ──────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    vocab_size:int=32000; dim:int=896; layers:int=20; heads:int=14
    kv_heads:int=7; hidden:int=2304; max_seq_len:int=2048; rope_theta:float=10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kh, self.d = c.heads, c.kv_heads, c.dim // c.heads
        self.q = nn.Linear(c.dim, self.h  * self.d, bias=False)
        self.k = nn.Linear(c.dim, self.kh * self.d, bias=False)
        self.v = nn.Linear(c.dim, self.kh * self.d, bias=False)
        self.o = nn.Linear(c.dim, c.dim,             bias=False)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, self.d, 2).float() / self.d))
        angles = torch.outer(torch.arange(c.max_seq_len).float(), inv)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)
    def rotate(self, x):
        cos = self.cos[:, :, :x.shape[2]].to(x.dtype)
        sin = self.sin[:, :, :x.shape[2]].to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a*cos - b*sin, a*sin + b*cos), dim=-1).flatten(-2)
    def forward(self, x):
        b, t, _ = x.shape
        q = self.rotate(self.q(x).view(b, t, self.h,  self.d).transpose(1, 2))
        k = self.rotate(self.k(x).view(b, t, self.kh, self.d).transpose(1, 2))
        v =             self.v(x).view(b, t, self.kh, self.d).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kh, dim=1)
        v = v.repeat_interleave(self.h // self.kh, dim=1)
        return self.o(F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                      .transpose(1, 2).contiguous().view(b, t, -1))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attention = Attention(c)
        self.gate = nn.Linear(c.dim, c.hidden, bias=False)
        self.up   = nn.Linear(c.dim, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.dim, bias=False)
    def forward(self, x):
        x = x + self.attention(self.n1(x))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))

class ScratchLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
    def forward(self, ids, last_only=False):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return F.linear(x[:, -1:] if last_only else x, self.embedding.weight)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(str(PILOT_DIR / "tokenizer.json"))

def encode_text(text):
    return tokenizer.encode(text).ids

PAD_ID       = tokenizer.token_to_id("<|pad|>")
EOS_ID       = tokenizer.token_to_id("<|eos|>")
SYSTEM_ID    = tokenizer.token_to_id("<|system|>")
USER_ID      = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")
BLOCKED_IDS  = [tid for tid in [
    PAD_ID, tokenizer.token_to_id("<|unk|>"),
    tokenizer.token_to_id("<|bos|>"),
    SYSTEM_ID, USER_ID, ASSISTANT_ID,
] if tid is not None]

SYSTEM_PROMPT = (SFT_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading scratch SFT v2 model on {DEVICE}...")
config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
model  = ScratchLM(config).to(DEVICE)
state  = torch.load(SFT_DIR / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
model.eval()
print(f"Loaded epoch {state['epoch']} | best_score={state.get('best_score','n/a')}")
del state

# ── Load RAG index ─────────────────────────────────────────────────────────────
print("Loading RAG index...")
idx_data = np.load(RAG_INDEX, allow_pickle=True)
rag_meta = json.loads(str(idx_data["metadata"]))
rag_docs  = rag_meta["documents"]

rag_chunks = []
for doc in rag_docs:
    embs = idx_data[doc["array_key"]]
    for i, chunk in enumerate(doc["chunks"]):
        rag_chunks.append({
            "text":           chunk,
            "emb":            embs[i],
            "filename":       doc["filename"],
            "student_folder": doc.get("student_folder", ""),
            "resume_id":      doc["resume_id"],
        })
all_embs = np.stack([c["emb"] for c in rag_chunks]).astype(np.float32)

print("Loading sentence encoder...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print(f"Ready — {len(rag_docs)} resumes, {len(rag_chunks)} chunks")

# ── Load scratch encoder + encoder index (optional — only if trained) ──────────
enc_model = None
enc_resume_embs = None
enc_resume_docs = None

if (ENC_DIR / "best.pt").exists() and ENC_INDEX.exists():
    try:
        from encoder_model import ScratchEncoder
        print("Loading scratch encoder...")
        ckpt = torch.load(ENC_DIR / "best.pt", map_location="cpu", weights_only=False)
        enc_model = ScratchEncoder(ckpt["config"]).to(DEVICE)
        enc_model.load_state_dict(ckpt["model"])
        enc_model.eval()
        enc_data = np.load(ENC_INDEX, allow_pickle=True)
        enc_meta = json.loads(str(enc_data["metadata"]))
        enc_resume_docs = enc_meta["documents"]
        enc_resume_embs = enc_data["embeddings"].astype(np.float32)
        print(f"Scratch encoder ready — {len(enc_resume_docs)} resumes, dim={enc_resume_embs.shape[1]}")
    except Exception as e:
        print(f"Scratch encoder not loaded: {e}")

ENC_MAX_LEN = 256

def encode_with_scratch(text: str) -> np.ndarray:
    ids = tokenizer.encode(text[:2000]).ids[:ENC_MAX_LEN]
    pad = ENC_MAX_LEN - len(ids)
    ids_t  = torch.tensor([ids + [PAD_ID] * pad], dtype=torch.long, device=DEVICE)
    mask_t = torch.tensor([[1]*len(ids) + [0]*pad], dtype=torch.long, device=DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        emb = enc_model(ids_t, mask_t).float().cpu().numpy()[0]
    return emb

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(context: str, question: str) -> list:
    context = context.strip() or "[No resume context was retrieved.]"
    return (
        [SYSTEM_ID]
        + encode_text("\n" + SYSTEM_PROMPT + "\n")
        + [USER_ID]
        + encode_text("\nResume context:\n" + context + "\n\nQuestion:\n" + question.strip() + "\n")
        + [ASSISTANT_ID]
        + encode_text("\n")
    )

def _rep_penalty(logits, ids, penalty=REPETITION_PENALTY):
    recent = set(ids[0, -80:].tolist())
    for tid in recent:
        if logits[0, tid] > 0:
            logits[0, tid] /= penalty
        else:
            logits[0, tid] *= penalty
    return logits

def _clean(text: str) -> str:
    import re
    text = re.sub(r'(\b[\w./ ]+)(,\s*\1)+', r'\1', text)
    return text.strip().rstrip(',').strip()

# ── classify_only: 1 forward pass, no loop — for fast multi-mode filtering ────
@torch.inference_mode()
def classify_only(context: str, question: str) -> str:
    prompt_ids = build_prompt(context, question)
    if len(prompt_ids) > model.config.max_seq_len - 1:
        prompt_ids = prompt_ids[:model.config.max_seq_len - 1]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids, last_only=True)[:, -1].float()
    logits[:, BLOCKED_IDS] = -float("inf")
    first_token = tokenizer.decode([logits.argmax(dim=-1).item()]).strip()
    return "FOUND" if first_token.upper().startswith("F") else "NOT_FOUND"

# ── generate: full greedy decode ──────────────────────────────────────────────
@torch.inference_mode()
def generate(context: str, question: str, max_new: int = MAX_NEW_TOKENS) -> dict:
    prompt_ids = build_prompt(context, question)
    if len(prompt_ids) > model.config.max_seq_len - max_new:
        prompt_ids = prompt_ids[:model.config.max_seq_len - max_new]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated_ids = []
    for _ in range(max_new):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids[:, -model.config.max_seq_len:], last_only=True)[:, -1].float()
        logits[:, BLOCKED_IDS] = -float("inf")
        logits = _rep_penalty(logits, ids)
        next_token = logits.argmax(dim=-1, keepdim=True)
        if next_token.item() == EOS_ID:
            break
        generated_ids.append(next_token.item())
        ids = torch.cat([ids, next_token], dim=1)
    raw = _clean(tokenizer.decode(generated_ids).strip())
    if re.match(r"^FOUND", raw, re.IGNORECASE):
        return {"status": "FOUND", "answer": re.sub(r"^FOUND\s*", "", raw, flags=re.IGNORECASE).strip()}
    elif re.match(r"^NOT_FOUND", raw, re.IGNORECASE):
        return {"status": "NOT_FOUND", "answer": "The requested information is not available in the provided context."}
    return {"status": "FOUND", "answer": raw}

# ── RAG retrieval ──────────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = 5):
    q_emb = encoder.encode([question], normalize_embeddings=True)[0].astype(np.float32)
    scores = all_embs @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(rag_chunks[i], float(scores[i])) for i in top_idx]

def retrieve_per_resume(question: str, top_n: int = 5, chunks_per: int = 2):
    q_emb = encoder.encode([question], normalize_embeddings=True)[0].astype(np.float32)
    scores = all_embs @ q_emb
    order  = np.argsort(scores)[::-1]
    seen = {}
    for idx in order:
        if len(seen) >= top_n:
            break
        chunk = rag_chunks[idx]
        rid = chunk["resume_id"]
        seen.setdefault(rid, [])
        if len(seen[rid]) < chunks_per:
            seen[rid].append((chunk, float(scores[idx])))
    results = []
    for pairs in seen.values():
        results.extend(pairs)
    return results

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Resume QA — Scratch SFT v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "single"

class CandidateResult(BaseModel):
    filename: str
    student_folder: str
    status: str
    answer: str
    context: str
    score: float

class AskResponse(BaseModel):
    status: str
    answer: str
    sources: list[str]
    context_used: str
    candidates: list[CandidateResult] = []

class MatchRequest(BaseModel):
    jd: str
    top_k: int = 5
    explain: bool = True   # use decoder to explain why each candidate matches

class MatchCandidate(BaseModel):
    filename: str
    student_folder: str
    score: float
    encoder_score: float
    rag_score: float
    explanation: str
    context: str

class MatchResponse(BaseModel):
    total_candidates: int
    candidates: list[MatchCandidate]

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE),
            "model": "scratch_sft_v2", "resumes_indexed": len(rag_docs)}

@app.get("/resumes")
def list_resumes():
    return {"total": len(rag_docs),
            "resumes": [{"filename": d["filename"],
                         "student_folder": d.get("student_folder", ""),
                         "resume_id": d["resume_id"]} for d in rag_docs]}

@app.get("/encoder_status")
def encoder_status():
    return {
        "scratch_encoder_loaded": enc_model is not None,
        "resumes_in_encoder_index": len(enc_resume_docs) if enc_resume_docs else 0,
    }

@app.post("/match", response_model=MatchResponse)
def match_jd(req: MatchRequest):
    """Match a Job Description against all resumes using encoder + RAG hybrid scoring."""
    if not req.jd.strip():
        raise HTTPException(status_code=400, detail="jd must not be empty")

    top_k = min(req.top_k, 10)

    # ── Step 1: RAG score (MiniLM cosine similarity on JD text) ────────────────
    rag_q_emb = encoder.encode([req.jd], normalize_embeddings=True)[0].astype(np.float32)
    rag_scores_all = all_embs @ rag_q_emb
    # Aggregate per resume: max chunk score
    rag_per_resume: dict = {}
    for i, chunk in enumerate(rag_chunks):
        rid = chunk["resume_id"]
        s = float(rag_scores_all[i])
        if rid not in rag_per_resume or s > rag_per_resume[rid]["score"]:
            rag_per_resume[rid] = {"score": s, "chunk": chunk}

    # ── Step 2: Encoder score (scratch encoder cosine similarity) ──────────────
    enc_per_resume: dict = {}
    if enc_model is not None and enc_resume_embs is not None:
        jd_emb = encode_with_scratch(req.jd)
        enc_scores = enc_resume_embs @ jd_emb
        for i, doc in enumerate(enc_resume_docs):
            enc_per_resume[doc["resume_id"]] = float(enc_scores[i])

    # ── Step 3: Hybrid score = 0.5 * RAG + 0.5 * encoder (or 1.0 * RAG if no encoder) ──
    all_rids = list(rag_per_resume.keys())
    hybrid_scores = []
    for rid in all_rids:
        rag_s = rag_per_resume[rid]["score"]
        enc_s = enc_per_resume.get(rid, rag_s)  # fallback to rag if encoder not loaded
        hybrid = 0.5 * rag_s + 0.5 * enc_s if enc_model is not None else rag_s
        hybrid_scores.append((rid, hybrid, rag_s, enc_s))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = hybrid_scores[:top_k]

    # ── Step 4: Decoder explanation for each top candidate ────────────────────
    candidates = []
    for rid, hybrid_s, rag_s, enc_s in top_results:
        chunk_info = rag_per_resume[rid]
        c = chunk_info["chunk"]
        explanation = ""
        if req.explain:
            question = f"Does this candidate match the following job description? {req.jd[:300]}"
            result = generate(c["text"], question, max_new=MAX_NEW_TOKENS)
            explanation = result["answer"]
        else:
            # Fast snippet
            text = c["text"].strip()
            m = re.search(r'[A-Z]', text)
            if m and m.start() < 80:
                text = text[m.start():]
            explanation = text[:200].strip()

        candidates.append(MatchCandidate(
            filename=c["filename"],
            student_folder=c["student_folder"],
            score=round(hybrid_s, 4),
            encoder_score=round(enc_s, 4),
            rag_score=round(rag_s, 4),
            explanation=explanation,
            context=c["text"],
        ))

    return MatchResponse(total_candidates=len(candidates), candidates=candidates)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    if req.mode == "multi":
        # Pure RAG: retrieve top-N resumes by embedding score, extract answer from chunk text
        pairs = retrieve_per_resume(req.question, top_n=req.top_k, chunks_per=2)
        by_resume: dict = {}
        for chunk, score in pairs:
            by_resume.setdefault(chunk["resume_id"], []).append((chunk, score))

        candidates = []
        for rid, chunk_pairs in by_resume.items():
            c0 = chunk_pairs[0][0]
            avg_score = sum(s for _, s in chunk_pairs) / len(chunk_pairs)
            # Extract a clean snippet from the best chunk (no model call)
            best_chunk = chunk_pairs[0][0]["text"].strip()
            # Find first capital letter to avoid mid-sentence starts
            import re
            match = re.search(r'[A-Z]', best_chunk)
            if match and match.start() < 80:
                best_chunk = best_chunk[match.start():]
            snippet = best_chunk[:250].strip()
            if len(best_chunk) > 250:
                last_sep = max(snippet.rfind('. '), snippet.rfind('\n'), snippet.rfind(', '))
                if last_sep > 80:
                    snippet = snippet[:last_sep + 1]
            candidates.append(CandidateResult(
                filename=c0["filename"], student_folder=c0["student_folder"],
                status="FOUND", answer=snippet,
                context="\n\n".join(c["text"] for c, _ in chunk_pairs),
                score=round(avg_score, 4)
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        summary = f"Found {len(candidates)} matching candidates."
        return AskResponse(status="FOUND", answer=summary,
                           sources=[c.filename for c in candidates],
                           context_used="", candidates=candidates)

    # Single mode
    pairs   = retrieve(req.question, top_k=req.top_k)
    context = "\n\n---\n\n".join(c["text"] for c, _ in pairs)
    result  = generate(context, req.question)
    sources = list(dict.fromkeys(c["filename"] for c, _ in pairs))
    return AskResponse(status=result["status"], answer=result["answer"],
                       sources=sources, context_used=context, candidates=[])
