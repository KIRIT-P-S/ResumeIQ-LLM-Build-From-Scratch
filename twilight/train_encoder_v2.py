"""
Encoder training v2 — external data + our data combined.
Data sources:
  - Our 9K synthetic resume pairs (3 JD variants each)
  - Our 440 real student resumes (4 JD variants each)
  - jacob-hugging-face/job-descriptions × resume-atlas cross pairs
  - resume-atlas category-matched pairs

Run: /opt/llm-training/bin/python3 train_encoder_v2.py
"""
import json, random, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from encoder_model import ScratchEncoder, EncoderConfig

ROOT     = Path.home() / "twilight"
EXT      = ROOT / "external_data"
PILOT    = ROOT / "scratch_resume_lm_pilot"
OUT_DIR  = ROOT / "scratch_encoder"
OUT_DIR.mkdir(exist_ok=True)

DEVICE      = torch.device("cuda:0")
EPOCHS      = 15
BATCH_SIZE  = 256      # bigger batch = more negatives = stronger signal
LR          = 1e-4
MAX_LEN     = 256
WARMUP_FRAC = 0.06
PATIENCE    = 4

print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(str(PILOT / "tokenizer.json"))
PAD_ID = tokenizer.token_to_id("<|pad|>") or 0

def encode(text, max_len=MAX_LEN):
    ids = tokenizer.encode(str(text)[:3000]).ids[:max_len]
    pad = max_len - len(ids)
    return ids + [PAD_ID] * pad, [1]*len(ids) + [0]*pad

# ── JD augmentation templates ──────────────────────────────────────────────────
TEMPLATES = [
    "We are hiring a {title}. {jd}",
    "Job Opening: {title}\n{jd}",
    "Position: {title}\nRequirements: {jd}",
    "{jd}",
    "Seeking a {title}. Key requirements: {jd}",
]

def augment_jd(jd: str, title: str = "") -> str:
    tmpl = random.choice(TEMPLATES)
    return tmpl.format(title=title or "professional", jd=jd[:800]).strip()

# ── Load all pairs ─────────────────────────────────────────────────────────────
print("Loading training pairs...")
pairs = []  # (jd_text, resume_text)

# 1. Our synthetic SFT data
resume_contexts = {}
with open(ROOT / "scratch_resume_sft_v2_data/train.jsonl") as f:
    for line in f:
        d = json.loads(line)
        rid = d.get("resume_id", "")
        if rid and d.get("context") and len(d["context"]) > 100:
            resume_contexts[rid] = d["context"]

JD_TEMPLATES_RICH = [
    "We are looking for a {role}. Required skills: {skills}. {exp}{edu}",
    "Job Opening: {role}\nMust have: {skills}\n{exp}{edu}",
    "Hiring: {role}. Key skills: {skills}. {exp}",
    "Position: {role}\nRequirements:\n- {skills_list}\n{exp}{edu}",
    "Seeking {role} with {skills}. {exp}{edu}",
]

def extract_and_make_jd(ctx, idx=0):
    skills, role, exp, edu = [], "", "", ""
    for line in ctx.split("\n"):
        l = line.strip()
        if any(l.startswith(k) for k in ["SKILLS","TECHNICAL","Languages","Programming"]) or "ACQUIRED" in l or "Tech Stack" in l:
            raw = re.sub(r'^[A-Z /]+[:\-]','',l).strip()
            skills += [s.strip() for s in re.split(r'[,|•\n]+', raw) if 2 < len(s.strip()) < 40]
        if l.startswith("ROLE"):
            role = re.sub(r'^ROLE\s*:?\s*','',l).strip()
        if l.startswith("YEARS"):
            exp = f"Experience: {re.sub(r'^YEARS.*?:','',l).strip()} years. "
        if l.startswith("EDUCATION") or "B.E" in l or "B.Tech" in l:
            edu = f"Education: {l[:60]}. "
    if not role: role = "Software Engineer"
    seen, clean = set(), []
    for s in skills:
        s = s.strip().strip(':-').strip()
        if s and s.lower() not in seen and len(s)>1:
            seen.add(s.lower()); clean.append(s)
    clean = clean[:10]
    if not clean: clean = ["programming","problem solving"]
    random.shuffle(clean)
    tmpl = JD_TEMPLATES_RICH[idx % len(JD_TEMPLATES_RICH)]
    return tmpl.format(
        role=role, skills=", ".join(clean[:7]),
        skills_list="\n- ".join(clean[:5]),
        exp=exp, edu=edu
    ).strip()

for rid, ctx in resume_contexts.items():
    for i in range(3):
        f2 = list(ctx)
        jd = extract_and_make_jd(ctx, i)
        pairs.append((jd, ctx[:1500]))

print(f"  Our synthetic pairs: {len(pairs)}")

# 2. Our real student resumes
rag_index = ROOT / "full_resume_rag_index.npz"
if rag_index.exists():
    idx_data = np.load(rag_index, allow_pickle=True)
    rag_meta = json.loads(str(idx_data["metadata"]))
    for doc in rag_meta["documents"]:
        chunks = doc.get("chunks", [])
        if not chunks: continue
        full_text = " ".join(chunks)
        for i in range(4):
            jd = extract_and_make_jd(full_text, i)
            pairs.append((jd, full_text[:1500]))
print(f"  After real resumes: {len(pairs)}")

# 3. External resume-atlas pairs
if (EXT / "encoder_pairs_resumes.jsonl").exists():
    with open(EXT / "encoder_pairs_resumes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["jd"], d["resume"]))
    print(f"  After resume-atlas: {len(pairs)}")

# 4. External cross pairs (JD × resume-atlas)
if (EXT / "encoder_pairs_cross.jsonl").exists():
    with open(EXT / "encoder_pairs_cross.jsonl") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["jd"], d["resume"]))
    print(f"  After cross pairs: {len(pairs)}")

# 5. Real JDs paired with resume-atlas resumes (augmented)
if (EXT / "job_descriptions.jsonl").exists() and (EXT / "resumes.jsonl").exists():
    jd_recs  = [json.loads(l) for l in open(EXT / "job_descriptions.jsonl")]
    res_recs = [json.loads(l) for l in open(EXT / "resumes.jsonl")]
    random.shuffle(jd_recs)
    random.shuffle(res_recs)
    # Pair each JD with a random resume (weak supervision — still useful)
    for i, jd_rec in enumerate(jd_recs[:5000]):
        res = res_recs[i % len(res_recs)]
        jd  = augment_jd(jd_rec["jd"], jd_rec.get("title",""))
        pairs.append((jd, res["text"][:1500]))
    print(f"  After real JD × resume-atlas: {len(pairs)}")

random.shuffle(pairs)
split       = int(len(pairs) * 0.92)
train_pairs = pairs[:split]
val_pairs   = pairs[split:]
print(f"\nTotal: {len(pairs)} pairs | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        jd, res = self.pairs[i]
        ji, jm = encode(jd)
        ri, rm = encode(res)
        return (torch.tensor(ji,dtype=torch.long), torch.tensor(jm,dtype=torch.long),
                torch.tensor(ri,dtype=torch.long), torch.tensor(rm,dtype=torch.long))

train_dl = DataLoader(PairDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)
val_dl   = DataLoader(PairDataset(val_pairs),   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

# ── NT-Xent loss ───────────────────────────────────────────────────────────────
def nt_xent(z1, z2, temp):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temp
    sim.masked_fill_(torch.eye(2*B, device=z.device).bool(), float('-inf'))
    labels = torch.cat([torch.arange(B,2*B,device=z.device), torch.arange(0,B,device=z.device)])
    return F.cross_entropy(sim, labels)

# ── Model — resume from best checkpoint ───────────────────────────────────────
print("\nBuilding encoder...")
config = EncoderConfig(vocab_size=tokenizer.get_vocab_size(), pad_id=PAD_ID)
model  = ScratchEncoder(config).to(DEVICE)

prev = OUT_DIR / "best.pt"
start_epoch = 1
if prev.exists():
    ckpt = torch.load(prev, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed from epoch {ckpt['epoch']}")
else:
    print(f"  Fresh — {sum(p.numel() for p in model.parameters()):,} params")

# ── LR: warmup + cosine ────────────────────────────────────────────────────────
total_steps  = EPOCHS * len(train_dl)
warmup_steps = int(total_steps * WARMUP_FRAC)
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9,0.98))

def lr_lambda(s):
    if s < warmup_steps: return s / max(1, warmup_steps)
    p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159*p)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Train ──────────────────────────────────────────────────────────────────────
best_val, no_improve, global_step = float('inf'), 0, 0

def get_temp(step, total):
    return max(0.04, 0.1 - 0.06 * (step / total))

print(f"Training {EPOCHS} epochs | batch={BATCH_SIZE} | {len(train_dl)} steps/epoch\n")

for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()
    total_loss = 0
    for ji, jm, ri, rm in tqdm(train_dl, desc=f"Epoch {epoch}"):
        ji,jm,ri,rm = ji.to(DEVICE),jm.to(DEVICE),ri.to(DEVICE),rm.to(DEVICE)
        temp = get_temp(global_step, total_steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = nt_xent(model(ji,jm), model(ri,rm), temp)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item(); global_step += 1

    train_loss = total_loss / len(train_dl)

    model.eval(); val_loss = 0
    with torch.no_grad():
        for ji, jm, ri, rm in val_dl:
            ji,jm,ri,rm = ji.to(DEVICE),jm.to(DEVICE),ri.to(DEVICE),rm.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                val_loss += nt_xent(model(ji,jm), model(ri,rm), 0.04).item()
    val_loss /= len(val_dl)

    print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | temp={get_temp(global_step,total_steps):.3f}")
    with open(OUT_DIR / "metrics_v2.jsonl", "a") as f:
        f.write(json.dumps({"epoch":epoch,"train":train_loss,"val":val_loss})+"\n")

    torch.save({"model":model.state_dict(),"config":config,"epoch":epoch}, OUT_DIR/"last.pt")
    if val_loss < best_val:
        best_val = val_loss; no_improve = 0
        torch.save({"model":model.state_dict(),"config":config,"epoch":epoch}, OUT_DIR/"best.pt")
        print(f"  ✓ Best saved (val={val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping (no improvement for {PATIENCE} epochs)")
            break

print(f"\nDone. Best val loss: {best_val:.4f}")
