"""
Decoder SFT v4 — fine-tune on external SQuAD + resume-atlas QA + our existing data.
Continues from SFT v2 best checkpoint.

Run: /opt/llm-training/bin/python3 train_decoder_v4.py
"""
import json, random, re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT     = Path.home() / "twilight"
EXT      = ROOT / "external_data"
PILOT    = ROOT / "scratch_resume_lm_pilot"
SFT_V2   = ROOT / "scratch_resume_lm_resume_sft_v2"
OUT_DIR  = ROOT / "scratch_resume_lm_resume_sft_v4"
OUT_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda:0")
EPOCHS     = 5
BATCH_SIZE = 8
GRAD_ACCUM = 4   # effective batch = 32
LR         = 5e-5
MAX_LEN    = 512
PATIENCE   = 2

# ── Model (same as app.py) ─────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    vocab_size:int=32000; dim:int=896; layers:int=20; heads:int=14
    kv_heads:int=7; hidden:int=2304; max_seq_len:int=2048; rope_theta:float=10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)).to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h,self.kh,self.d = c.heads,c.kv_heads,c.dim//c.heads
        self.q = nn.Linear(c.dim,self.h*self.d,bias=False)
        self.k = nn.Linear(c.dim,self.kh*self.d,bias=False)
        self.v = nn.Linear(c.dim,self.kh*self.d,bias=False)
        self.o = nn.Linear(c.dim,c.dim,bias=False)
        inv = 1.0/(c.rope_theta**(torch.arange(0,self.d,2).float()/self.d))
        angles = torch.outer(torch.arange(c.max_seq_len).float(),inv)
        self.register_buffer("cos",angles.cos()[None,None],persistent=False)
        self.register_buffer("sin",angles.sin()[None,None],persistent=False)
    def rotate(self,x):
        cos=self.cos[:,:,:x.shape[2]].to(x.dtype); sin=self.sin[:,:,:x.shape[2]].to(x.dtype)
        a,b=x[...,0::2],x[...,1::2]
        return torch.stack((a*cos-b*sin,a*sin+b*cos),dim=-1).flatten(-2)
    def forward(self,x):
        b,t,_=x.shape
        q=self.rotate(self.q(x).view(b,t,self.h,self.d).transpose(1,2))
        k=self.rotate(self.k(x).view(b,t,self.kh,self.d).transpose(1,2))
        v=self.v(x).view(b,t,self.kh,self.d).transpose(1,2)
        k=k.repeat_interleave(self.h//self.kh,dim=1); v=v.repeat_interleave(self.h//self.kh,dim=1)
        return self.o(F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True).transpose(1,2).contiguous().view(b,t,-1))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1,self.n2=RMSNorm(c.dim),RMSNorm(c.dim)
        self.attention=Attention(c)
        self.gate=nn.Linear(c.dim,c.hidden,bias=False)
        self.up=nn.Linear(c.dim,c.hidden,bias=False)
        self.down=nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x):
        x=x+self.attention(self.n1(x)); y=self.n2(x)
        return x+self.down(F.silu(self.gate(y))*self.up(y))

class ScratchLM(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.dim)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm=RMSNorm(config.dim)
    def forward(self,ids):
        x=self.embedding(ids)
        for block in self.blocks: x=block(x)
        return F.linear(self.norm(x),self.embedding.weight)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer  = Tokenizer.from_file(str(PILOT/"tokenizer.json"))
PAD_ID     = tokenizer.token_to_id("<|pad|>")
EOS_ID     = tokenizer.token_to_id("<|eos|>")
SYSTEM_ID  = tokenizer.token_to_id("<|system|>")
USER_ID    = tokenizer.token_to_id("<|user|>")
ASST_ID    = tokenizer.token_to_id("<|assistant|>")

SYSTEM_PROMPT = (SFT_V2/"system_prompt.txt").read_text().strip()

def enc(text): return tokenizer.encode(text).ids

def build_ids(context, question, answer):
    prompt = (
        [SYSTEM_ID] + enc("\n"+SYSTEM_PROMPT+"\n") +
        [USER_ID]   + enc("\nResume context:\n"+context+"\n\nQuestion:\n"+question+"\n") +
        [ASST_ID]   + enc("\n")
    )
    response = enc(answer) + [EOS_ID]
    return prompt, response

# ── Load all SFT data ──────────────────────────────────────────────────────────
print("Loading SFT data...")
all_examples = []

def add_from_jsonl(path, context_key="context", question_key="question", answer_key="answer", limit=None):
    count = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ctx = d.get(context_key,"").strip()
            q   = d.get(question_key,"").strip()
            ans = d.get(answer_key,"").strip()
            if not ctx or not q or not ans: continue
            answerable = d.get("answerable", True)
            if answerable and ans.upper() != "NOT_FOUND":
                formatted_ans = f"FOUND {ans}"
            else:
                formatted_ans = "NOT_FOUND"
            all_examples.append((ctx[:800], q, formatted_ans))
            count += 1
            if limit and count >= limit: break
    return count

# Our existing SFT v2 data
n = add_from_jsonl(ROOT/"scratch_resume_sft_v2_data/train.jsonl", limit=30000)
print(f"  SFT v2 train: {n}")

# SQuAD external data — reading comprehension teaches the model to extract answers
if (EXT/"squad_sft.jsonl").exists():
    n = add_from_jsonl(EXT/"squad_sft.jsonl", limit=20000)
    print(f"  SQuAD: {n}")

# Resume-atlas QA — generate QA from resume categories
if (EXT/"resumes.jsonl").exists():
    RESUME_QUESTIONS = [
        ("What is the candidate's professional category?", "category"),
        ("What skills does this candidate have?", None),
        ("Summarize this candidate's background.", None),
    ]
    with open(EXT/"resumes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            cat  = d.get("category","")
            text = d.get("text","").strip()
            if not text: continue
            # Category question
            all_examples.append((text[:800], "What is the candidate's professional category or role?", f"FOUND {cat}"))
            # Skills question
            skills_match = re.findall(r'\b(?:Python|Java|SQL|React|Node|AWS|Docker|ML|Excel|C\+\+|JavaScript|Kubernetes|TensorFlow)\b', text)
            if skills_match:
                all_examples.append((text[:800], "What technical skills does this candidate have?", f"FOUND {', '.join(set(skills_match[:8]))}"))
    print(f"  Resume-atlas QA: added, total={len(all_examples)}")

random.shuffle(all_examples)
split = int(len(all_examples)*0.92)
train_ex = all_examples[:split]
val_ex   = all_examples[split:]
print(f"\nTotal: {len(all_examples)} | Train: {len(train_ex)} | Val: {len(val_ex)}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, examples, max_len=MAX_LEN):
        self.examples = examples
        self.max_len  = max_len

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ctx, q, ans = self.examples[i]
        prompt, response = build_ids(ctx, q, ans)
        full = (prompt + response)[:self.max_len]
        labels = [-100]*len(prompt) + response
        labels = labels[:self.max_len]
        pad = self.max_len - len(full)
        input_ids = full + [PAD_ID]*pad
        labels    = labels + [-100]*pad
        return torch.tensor(input_ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long)

train_dl = DataLoader(SFTDataset(train_ex), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_dl   = DataLoader(SFTDataset(val_ex),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Load model from SFT v2 ─────────────────────────────────────────────────────
print("\nLoading SFT v2 model...")
config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
model  = ScratchLM(config).to(DEVICE)
state  = torch.load(SFT_V2/"best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
print(f"  Loaded epoch {state['epoch']} | best_score={state.get('best_score','n/a')}")
del state

# Copy system prompt
import shutil
shutil.copy(SFT_V2/"system_prompt.txt", OUT_DIR/"system_prompt.txt")

# ── Optimizer: lower LR to avoid forgetting ───────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = EPOCHS * (len(train_dl) // GRAD_ACCUM)
warmup_steps = int(total_steps * 0.05)

def lr_lambda(s):
    if s < warmup_steps: return s/max(1,warmup_steps)
    p = (s-warmup_steps)/max(1,total_steps-warmup_steps)
    return 0.1 + 0.9*0.5*(1+torch.cos(torch.tensor(3.14159*p)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Training ───────────────────────────────────────────────────────────────────
best_val, no_improve, sched_step = float('inf'), 0, 0
print(f"\nTraining {EPOCHS} epochs | batch={BATCH_SIZE} | grad_accum={GRAD_ACCUM} | effective_batch={BATCH_SIZE*GRAD_ACCUM}\n")

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0; optimizer.zero_grad()
    for step, (ids, labels) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids)
            loss   = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)
            loss   = loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM
        if (step+1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            sched_step += 1

    train_loss = total_loss / len(train_dl)

    model.eval(); val_loss = 0
    with torch.no_grad():
        for ids, labels in val_dl:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
                val_loss += F.cross_entropy(logits.view(-1,config.vocab_size), labels.view(-1), ignore_index=-100).item()
    val_loss /= len(val_dl)

    print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
    with open(OUT_DIR/"metrics.jsonl","a") as f:
        f.write(json.dumps({"epoch":epoch,"train":train_loss,"val":val_loss})+"\n")

    torch.save({"model":model.state_dict(),"epoch":epoch,"val_loss":val_loss}, OUT_DIR/"last.pt")
    if val_loss < best_val:
        best_val = val_loss; no_improve = 0
        torch.save({"model":model.state_dict(),"epoch":epoch,"val_loss":val_loss,"best_score":val_loss}, OUT_DIR/"best.pt")
        print(f"  ✓ Best saved (val={val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping"); break

print(f"\nDone. Best val loss: {best_val:.4f}")
print(f"Checkpoint: {OUT_DIR/'best.pt'}")
