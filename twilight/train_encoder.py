"""
Stronger encoder training:
- Richer JD generation (multi-sentence, varied templates)
- Multiple JD augmentations per resume (3x data)
- Larger batch (128) = more in-batch negatives
- Hard negatives: same-skill resumes as negatives
- Warmup + cosine LR schedule
- 10 epochs with early stopping
- Temperature annealing (start 0.1 → end 0.05)

Run: /opt/llm-training/bin/python3 train_encoder.py
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

ROOT      = Path.home() / "twilight"
PILOT_DIR = ROOT / "scratch_resume_lm_pilot"
OUT_DIR   = ROOT / "scratch_encoder"
OUT_DIR.mkdir(exist_ok=True)

DEVICE      = torch.device("cuda:0")
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 2e-4
MAX_LEN     = 256
WARMUP_FRAC = 0.1
PATIENCE    = 3   # early stopping

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(str(PILOT_DIR / "tokenizer.json"))
PAD_ID = tokenizer.token_to_id("<|pad|>") or 0

def encode(text, max_len=MAX_LEN):
    ids = tokenizer.encode(str(text)[:3000]).ids[:max_len]
    pad = max_len - len(ids)
    return ids + [PAD_ID] * pad, [1] * len(ids) + [0] * pad

# ── Rich JD generation ─────────────────────────────────────────────────────────
JD_TEMPLATES = [
    "We are hiring a {role}. Required skills: {skills}. {exp_line}{edu_line}",
    "Job Opening: {role}\nResponsibilities: Work with {skills}.\n{exp_line}We prefer candidates with {edu_line}",
    "Position: {role}\nMust have: {skills}\n{exp_line}Qualifications: {edu_line}",
    "Seeking a talented {role} to join our team. You should be proficient in {skills}. {exp_line}{edu_line}",
    "Role: {role}\nKey Requirements:\n- Strong knowledge of {skills_list}\n{exp_line}{edu_line}",
    "We need a {role} with hands-on experience in {skills}. {exp_line}",
    "{role} needed. Skills: {skills}. {exp_line}{edu_line}",
]

def extract_fields(ctx: str):
    skills, role, exp, edu, projects, certs = [], "", "", "", [], []
    lines = ctx.split("\n")
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Skills extraction — many formats
        if any(l.startswith(k) for k in ["SKILLS", "TECHNICAL SKILLS", "Languages", "Programming"]):
            raw = re.sub(r'^[A-Z /]+[:\-]', '', l).strip()
            skills += [s.strip() for s in re.split(r'[,|•\n]+', raw) if len(s.strip()) > 1]
        if "ACQUIRED" in l or "Tech Stack" in l or "tech stack" in l:
            raw = re.sub(r'.*?:', '', l).strip()
            skills += [s.strip() for s in re.split(r'[,|•]+', raw) if len(s.strip()) > 1]
        if l.startswith("ROLE"):
            role = re.sub(r'^ROLE\s*:?\s*', '', l).strip()
        if l.startswith("YEARS OF EXPERIENCE"):
            exp = re.sub(r'^YEARS OF EXPERIENCE\s*:?\s*', '', l).strip()
        if l.startswith("EDUCATION") or "B.E" in l or "B.Tech" in l or "BSc" in l:
            edu = l[:80]
        if l.startswith("PROJECT") or "project" in l.lower()[:20]:
            projects.append(l[:60])
        if "CERTIF" in l.upper():
            certs.append(l[:60])
    # Deduplicate skills
    seen = set()
    clean_skills = []
    for s in skills:
        s = s.strip().strip(':-').strip()
        if s and s.lower() not in seen and len(s) > 1 and len(s) < 40:
            seen.add(s.lower())
            clean_skills.append(s)
    return {
        "role": role or "Software Engineer",
        "skills": clean_skills[:12],
        "exp": exp,
        "edu": edu,
        "projects": projects[:3],
        "certs": certs[:2],
    }

def make_jd(fields: dict, template_idx: int = None) -> str:
    skills = fields["skills"]
    if not skills:
        skills = ["programming", "problem solving"]
    random.shuffle(skills)
    skills_str = ", ".join(skills[:8])
    skills_list = "\n- ".join(skills[:6])
    exp_line = f"Experience: {fields['exp']} years. " if fields["exp"] else ""
    edu_line = f"Education: {fields['edu']}. " if fields["edu"] else ""
    tmpl = JD_TEMPLATES[template_idx % len(JD_TEMPLATES)] if template_idx is not None else random.choice(JD_TEMPLATES)
    jd = tmpl.format(
        role=fields["role"], skills=skills_str,
        skills_list=skills_list, exp_line=exp_line, edu_line=edu_line
    )
    # Optionally append project/cert context
    if fields["projects"] and random.random() > 0.5:
        jd += f" Experience with projects like: {fields['projects'][0]}."
    return jd.strip()

def make_jd_variants(ctx: str, n: int = 3):
    """Generate n different JD variants from one resume — data augmentation."""
    fields = extract_fields(ctx)
    jds = []
    for i in range(n):
        # Randomly drop some skills to create partial-match JDs
        f = dict(fields)
        if len(f["skills"]) > 3:
            keep = random.randint(max(2, len(f["skills"]) // 2), len(f["skills"]))
            f["skills"] = random.sample(f["skills"], keep)
        jds.append(make_jd(f, template_idx=i))
    return jds

# ── Build pairs ────────────────────────────────────────────────────────────────
print("Building JD-Resume pairs...")

pairs = []  # (jd_text, resume_text, resume_id)

# From SFT v2 synthetic data
resume_contexts = {}
with open(ROOT / "scratch_resume_sft_v2_data/train.jsonl") as f:
    for line in f:
        d = json.loads(line)
        rid = d.get("resume_id", "")
        if rid and d.get("context") and len(d["context"]) > 100:
            resume_contexts[rid] = d["context"]

print(f"  Synthetic resumes: {len(resume_contexts)}")
for rid, ctx in resume_contexts.items():
    for jd in make_jd_variants(ctx, n=3):
        pairs.append((jd, ctx[:1500], rid))

# From real resumes via RAG index
rag_index = ROOT / "full_resume_rag_index.npz"
if rag_index.exists():
    idx_data = np.load(rag_index, allow_pickle=True)
    rag_meta = json.loads(str(idx_data["metadata"]))
    for doc in rag_meta["documents"]:
        chunks = doc.get("chunks", [])
        if not chunks:
            continue
        full_text = " ".join(chunks)
        rid = doc["resume_id"]
        for jd in make_jd_variants(full_text, n=4):  # more variants for real resumes
            pairs.append((jd, full_text[:1500], rid))

print(f"  Total pairs: {len(pairs)}")

random.shuffle(pairs)
split = int(len(pairs) * 0.9)
train_pairs = pairs[:split]
val_pairs   = pairs[split:]
print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")

# ── Dataset with hard negatives ────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        jd, resume, _ = self.pairs[i]
        jd_ids,  jd_mask  = encode(jd)
        res_ids, res_mask = encode(resume)
        return (
            torch.tensor(jd_ids,   dtype=torch.long),
            torch.tensor(jd_mask,  dtype=torch.long),
            torch.tensor(res_ids,  dtype=torch.long),
            torch.tensor(res_mask, dtype=torch.long),
        )

train_ds = PairDataset(train_pairs)
val_ds   = PairDataset(val_pairs)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── NT-Xent with temperature annealing ────────────────────────────────────────
def nt_xent_loss(z1, z2, temp):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temp
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)

# ── Model — load from previous checkpoint if exists ───────────────────────────
print("Building encoder...")
config = EncoderConfig(vocab_size=tokenizer.get_vocab_size(), pad_id=PAD_ID)
model  = ScratchEncoder(config).to(DEVICE)

prev_ckpt = OUT_DIR / "best.pt"
start_epoch = 1
if prev_ckpt.exists():
    ckpt = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed from epoch {ckpt['epoch']} (val_loss from prev run)")
else:
    print(f"  Fresh start — {sum(p.numel() for p in model.parameters()):,} params")

# ── LR schedule: linear warmup + cosine decay ─────────────────────────────────
total_steps  = EPOCHS * len(train_dl)
warmup_steps = int(total_steps * WARMUP_FRAC)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Training ───────────────────────────────────────────────────────────────────
best_val_loss = float('inf')
no_improve    = 0
global_step   = 0

# Temperature annealing: start at 0.1, end at 0.05
def get_temp(step, total):
    return 0.1 - 0.05 * (step / total)

print(f"\nTraining {EPOCHS} epochs | batch={BATCH_SIZE} | {len(train_dl)} steps/epoch")
print(f"Warmup: {warmup_steps} steps | Total: {total_steps} steps\n")

for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()
    total_loss = 0
    for jd_ids, jd_mask, res_ids, res_mask in tqdm(train_dl, desc=f"Epoch {epoch}"):
        jd_ids,  jd_mask  = jd_ids.to(DEVICE),  jd_mask.to(DEVICE)
        res_ids, res_mask = res_ids.to(DEVICE), res_mask.to(DEVICE)
        temp = get_temp(global_step, total_steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_jd  = model(jd_ids,  jd_mask)
            z_res = model(res_ids, res_mask)
            loss  = nt_xent_loss(z_jd, z_res, temp)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss  += loss.item()
        global_step += 1

    train_loss = total_loss / len(train_dl)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for jd_ids, jd_mask, res_ids, res_mask in val_dl:
            jd_ids,  jd_mask  = jd_ids.to(DEVICE),  jd_mask.to(DEVICE)
            res_ids, res_mask = res_ids.to(DEVICE), res_mask.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_jd  = model(jd_ids,  jd_mask)
                z_res = model(res_ids, res_mask)
                loss  = nt_xent_loss(z_jd, z_res, 0.05)
            val_loss += loss.item()
    val_loss /= len(val_dl)

    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f} | lr={lr_now:.2e} | temp={get_temp(global_step,total_steps):.3f}")

    with open(OUT_DIR / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}) + "\n")

    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, OUT_DIR / "last.pt")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, OUT_DIR / "best.pt")
        print(f"  ✓ Best saved (val={val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nDone. Best val loss: {best_val_loss:.4f}")
