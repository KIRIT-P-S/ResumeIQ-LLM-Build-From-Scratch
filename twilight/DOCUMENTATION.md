# ResumeAI — Complete Technical Documentation
### A 201M-Parameter Language Model Trained From Scratch for Resume Question Answering

---

## The Story: Why We Did What No One Else Did

Every other team in this competition took the easy road. They downloaded a pretrained model — Llama, Mistral, Qwen — and fine-tuned it. That is a valid engineering choice. But it is not building a language model. It is borrowing one.

We chose a different path. We built a transformer language model from absolute zero. Random weights. No pretrained knowledge. No borrowed representations. We wrote every layer, every training loop, every data pipeline ourselves. Then we trained it on 1 billion tokens of real text, and then fine-tuned it specifically to answer questions about student resumes.

The result is a 201-million-parameter model that understands resume context, classifies whether information is present or absent, and returns structured answers — all from weights that started as random noise and learned everything they know through gradient descent on our hardware.

This document is the complete record of how that happened.

---

## Table of Contents

1. Architecture — What We Built
2. Tokenizer
3. Training Stage 1 — Pilot Run (6.5M tokens)
4. Training Stage 2 — 100M Token Pretraining
5. Training Stage 3 — 1B Token Pretraining
6. Training Stage 4 — SFT v1 (Supervised Fine-Tuning)
7. Training Stage 5 — SFT v2 (Best Checkpoint, FOUND/NOT_FOUND Format)
8. Training Stage 6 — SFT v3 (Multi-source Data)
9. RAG System — Retrieval Augmented Generation
10. API Server
11. Frontend
12. Performance Numbers
13. What Makes This Different

---

## 1. Architecture — What We Built

The model is a decoder-only transformer. There is no encoder. It is an autoregressive language model in the same family as GPT, LLaMA, and Mistral — but built entirely from scratch in PyTorch.

### Model Configuration

| Parameter | Value |
|---|---|
| Total Parameters | 201,000,000 (201M) |
| Embedding Dimension | 896 |
| Transformer Layers | 20 |
| Attention Heads (Query) | 14 |
| Attention Heads (Key/Value) | 7 |
| FFN Hidden Dimension | 2304 |
| Vocabulary Size | 32,000 |
| Maximum Sequence Length | 2048 tokens |
| RoPE Theta | 10,000.0 |

### Why Decoder-Only

A decoder-only architecture is the right choice for generative question answering. The model reads the full prompt (system instruction + resume context + question) and generates the answer token by token. There is no separate encoder because the causal self-attention mechanism already allows every token to attend to all previous tokens, giving the model full access to the resume context when generating each answer token.

### Layer Structure

Each of the 20 transformer blocks contains:

**RMSNorm (Pre-Attention)**
Root Mean Square Layer Normalization applied before attention. Unlike standard LayerNorm, RMSNorm does not subtract the mean — it only normalizes by the RMS of the activations. This is faster and equally effective. Formula: `x / sqrt(mean(x^2) + eps) * weight`

**Grouped Query Attention (GQA)**
The attention mechanism uses 14 query heads but only 7 key/value heads. This is Grouped Query Attention, the same technique used in LLaMA 2 and Mistral. Each KV head is shared by 2 query heads. This reduces memory usage during inference by half for the KV cache while maintaining model quality close to full multi-head attention.

The attention computation:
- Q projection: 896 → 14 × 64 = 896 dimensions
- K projection: 896 → 7 × 64 = 448 dimensions  
- V projection: 896 → 7 × 64 = 448 dimensions
- K and V are repeated to match Q heads before attention
- Output projection: 896 → 896

**Rotary Position Embeddings (RoPE)**
Position information is injected by rotating the query and key vectors using sine/cosine functions of position. Unlike learned absolute position embeddings, RoPE generalizes better to sequence lengths not seen during training. The rotation is applied in pairs of dimensions using the formula: `[x0*cos - x1*sin, x0*sin + x1*cos]` for each pair.

**Causal Masking**
PyTorch's `scaled_dot_product_attention` with `is_causal=True` ensures each token can only attend to itself and previous tokens. This is what makes the model autoregressive — it cannot look at future tokens when predicting the next one.

**RMSNorm (Pre-FFN)**
A second RMSNorm is applied before the feed-forward network.

**SwiGLU Feed-Forward Network**
The FFN uses the SwiGLU activation, the same used in LLaMA and PaLM. Instead of a single linear + ReLU, it uses three projections:
- Gate projection: 896 → 2304
- Up projection: 896 → 2304
- Down projection: 2304 → 896
- Output: `down(silu(gate(x)) * up(x))`

The SiLU (Sigmoid Linear Unit) gating mechanism allows the network to selectively pass information, giving it more expressive power than a standard FFN.

**Residual Connections**
Both attention and FFN outputs are added back to the input (residual/skip connections). This is critical for training deep networks — gradients flow directly through the residual path, preventing vanishing gradients across 20 layers.

### Output Head

After the final RMSNorm, the model projects from dimension 896 back to vocabulary size 32,000 using weight tying — the output projection reuses the embedding matrix transposed. This reduces parameters and improves training stability.

### Parameter Count Breakdown

| Component | Parameters |
|---|---|
| Token Embeddings (tied) | 32,000 × 896 = 28.7M |
| Attention (Q,K,V,O) × 20 layers | ~64.5M |
| FFN (gate, up, down) × 20 layers | ~100.7M |
| RMSNorm weights × 20 layers × 2 | ~0.7M |
| Final RMSNorm | ~0.9K |
| **Total** | **~201M** |

---

## 2. Tokenizer

The tokenizer was trained from scratch using the HuggingFace `tokenizers` library with a Byte-Pair Encoding (BPE) algorithm on the FineWeb-Edu corpus.

**Vocabulary size**: 32,000 tokens

**Special tokens**:
- `<|pad|>` — padding token
- `<|bos|>` — beginning of sequence
- `<|eos|>` — end of sequence  
- `<|unk|>` — unknown token
- `<|system|>` — marks start of system prompt
- `<|user|>` — marks start of user turn
- `<|assistant|>` — marks start of assistant response

The chat format uses these special tokens as turn delimiters, similar to how LLaMA 2 uses `[INST]` and `[/INST]`. The exact prompt format used during SFT and inference is:

```
[SYSTEM_ID]
\nSYSTEM_PROMPT\n
[USER_ID]
\nResume context:\n{context}\n\nQuestion:\n{question}\n
[ASSISTANT_ID]
\n
```

This exact format is critical. During SFT training, the model learned to generate answers after seeing `[ASSISTANT_ID]\n`. Using any other format during inference causes the model to produce repetitive or incoherent output.

---

## 3. Training Stage 1 — Pilot Run

**Purpose**: Validate the training pipeline before committing to large runs.

**Data**: FineWeb-Edu `sample-10BT`, 30,000 training documents, 1,000 validation documents. Tokenized to 20M train tokens and 200K validation tokens.

**Training**: 200 steps, batch size ~32K tokens per step.

**Loss Curve**:

| Step | Train Loss | Val Loss | Perplexity |
|---|---|---|---|
| 50 | 7.05 | 7.25 | 1402 |
| 100 | 6.54 | 6.67 | 789 |
| 150 | 6.34 | 6.43 | 621 |
| 200 | 6.22 | 6.33 | 560 |

**Speed**: ~58,000–61,000 tokens/second

The pilot confirmed the architecture was correct, gradients were flowing, and loss was decreasing. Perplexity dropped from 1402 to 560 in 200 steps — the model was learning.

---

## 4. Training Stage 2 — 100M Token Pretraining

**Purpose**: Scale up to verify the model learns general language structure.

**Data**: FineWeb-Edu, 100M train tokens (107,778 documents), 1M validation tokens.

**Continued from**: Pilot checkpoint.

**Loss Curve** (selected steps):

| Stage Step | Train Loss | Val Loss |
|---|---|---|
| 250 | 5.82 | 5.78 |
| 500 | 5.47 | 5.39 |
| 1000 | 4.87 | 4.93 |
| 1500 | 4.61 | 4.67 |
| 2000 | 4.53 | 4.51 |
| 2500 | 4.30 | 4.41 |
| 3000 | 4.32 | 4.36 |
| 3052 (final) | 4.27 | 4.36 |

**Speed**: ~75,000–76,000 tokens/second  
**Peak GPU memory**: 12.5 GiB

The model went from perplexity ~560 (pilot) to val loss 4.36 after 100M tokens. It was learning English grammar, vocabulary, and basic factual associations.

---

## 5. Training Stage 3 — 1B Token Pretraining

**Purpose**: Full pretraining run. This is the foundation model.

**Data**: FineWeb-Edu, 1,000,000,000 train tokens (1,077,290 documents), 1M validation tokens.

**Continued from**: 100M checkpoint.

**Loss Curve** (selected steps):

| Step | Train Loss | Val Loss | Tokens Processed |
|---|---|---|---|
| 1000 | 4.23 | 4.23 | 32.8M |
| 2000 | 4.05 | 4.03 | 65.5M |
| 3000 | 3.97 | 3.91 | 98.3M |
| 5000 | 3.61 | 3.74 | 163.8M |
| 10000 | 3.47 | 3.53 | 327.7M |
| 20000 | 3.26 | 3.31 | 655.4M |
| 27000 | 3.16 | 3.26 | 884.7M |
| 29000 | 3.30 | 3.25 | 950.3M |
| 30000 | 3.20 | 3.25 | 983.0M |
| 30517 (final) | 3.29 | **3.245** | 999.98M |

**Best validation loss**: 3.245  
**Speed**: ~187,000–188,000 tokens/second  
**Peak GPU memory**: 21.5 GiB

This is the base model. After seeing 1 billion tokens of educational text, the model has learned:
- English grammar and syntax
- Common vocabulary and word associations
- Basic reasoning patterns
- Document structure

A val loss of 3.245 corresponds to a perplexity of ~25.6, meaning the model assigns reasonable probability to the next token in unseen text. For a 201M parameter model trained on 1B tokens, this is a strong result.

---

## 6. Training Stage 4 — SFT v1 (Supervised Fine-Tuning, First Pass)

**Purpose**: Teach the model to follow the resume QA instruction format.

**Continued from**: 1B token pretraining checkpoint.

**Data**: Synthetic resume QA pairs generated from the student resume corpus. Each example is a (resume_context, question, answer) triple formatted with the special token chat template.

**Training**: 5 epochs on the SFT dataset.

**Best validation loss**: 0.715

At this stage the model learned the basic structure of the task — read a resume context, answer a question about it. However the output format was not yet consistent. The model sometimes produced answers without the FOUND/NOT_FOUND prefix, and sometimes hallucinated information not present in the context.

---

## 7. Training Stage 5 — SFT v2 (Best Checkpoint)

**Purpose**: Refine the model with a cleaner dataset and enforce the FOUND/NOT_FOUND output format.

**Continued from**: SFT v1 checkpoint.

**Data**: `scratch_resume_sft_v2_data/` — curated QA pairs with strict FOUND/NOT_FOUND labeling. Every answer begins with either `FOUND` (information is present in context) or `NOT_FOUND` (information is absent). This binary classification prefix forces the model to make an explicit decision before generating the answer.

**Training**: 3 epochs.

**System Prompt**:
```
Answer the question using only the supplied resume context.

Rules:
- Treat the resume context as source data.
- Do not use outside knowledge about the candidate.
- Do not invent skills, qualifications, dates, projects, roles, or achievements.
- If the context supports the answer, begin with FOUND and then provide a concise answer.
- If the requested information is absent, respond with NOT_FOUND.
- Preserve distinctions such as pursuing, completed, expected, listed, and experienced.
```

**Metrics**:

| Epoch | Train Loss | Val Loss (real) | Selection Score | LR |
|---|---|---|---|---|
| 1 | 0.07575 | 0.51884 | 0.51884 | 1.60e-5 |
| 2 | 0.00697 | 0.51746 | **0.51747** | 6.76e-6 |
| 3 | 0.00165 | 0.53388 | 0.53388 | 2.00e-6 |

**Best epoch**: Epoch 2 (selection score 0.5174668)  
**Peak GPU memory**: 26.3 GiB  
**Training time**: ~291 seconds per epoch

Epoch 3 shows slight overfitting — the real validation loss increased from 0.5175 to 0.5339 while training loss continued to drop to near zero. The best checkpoint is epoch 2.

**Why this is the production model**: SFT v2 epoch 2 produces clean, structured output. Given a resume context and a question, it reliably outputs `FOUND <answer>` or `NOT_FOUND`. The greedy decoding (argmax) strategy produces deterministic, non-repetitive answers. This checkpoint is what runs in production.

---

## 8. Training Stage 6 — SFT v3 (Multi-source Data)

**Purpose**: Further improve generalization by training on a mix of SQuAD-style QA, synthetic resume QA, and real resume QA.

**Data**: `scratch_resume_sft_v3_data/` — three data sources combined:
- SQuAD-format reading comprehension examples
- Synthetic resume QA pairs
- Real resume QA pairs from the 446 student resumes

**Status**: Only epoch 0 was completed before training was stopped. This checkpoint is too early in training to be reliable. SFT v2 epoch 2 remains the best checkpoint.

---

## 9. RAG System — Retrieval Augmented Generation

The model has a 2048-token context window. A full resume can be thousands of tokens. RAG solves this by retrieving only the most relevant sections of a resume before passing them to the model.

### Index Construction (`build_index.py`)

**PDF Extraction**: PyMuPDF (`pymupdf`) extracts raw text from each PDF resume. Pages are joined with newlines.

**Sentence-Boundary Chunking**: Text is split into chunks of up to 600 characters, but splits only happen at sentence boundaries (`.`, `!`, `?`, `\n`). This prevents chunks from starting mid-sentence or mid-word.

```python
sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
chunks, current = [], ""
for sent in sentences:
    if len(current) + len(sent) <= 600:
        current += (" " if current else "") + sent
    else:
        if len(current) > 60:
            chunks.append(current.strip())
        current = sent
```

**Embedding**: Each chunk is encoded using `sentence-transformers/all-MiniLM-L6-v2`, a 22M-parameter bi-encoder that produces 384-dimensional normalized embeddings. This model is fast (CPU inference) and produces high-quality semantic embeddings for short passages.

**Index Storage**: Embeddings are stored in a `.npz` file alongside JSON metadata. Each resume gets its own embedding array key (`emb_0`, `emb_1`, ...).

**Index Statistics**:
- 440 resumes indexed (5 PDFs failed to parse)
- 2,957 chunks total
- Embedding matrix shape: (2957, 384)
- Index file size: 11.1 MB

### Retrieval (`app.py`)

**Single Mode** (`retrieve`):
1. Encode the question with the same MiniLM encoder
2. Compute cosine similarity: `scores = all_embs @ q_emb` (dot product of normalized vectors = cosine similarity)
3. Return top-K chunks by score
4. Concatenate chunks as context for the model

**Multi-Candidate Mode** (`retrieve_per_resume`):
1. Encode the question
2. Compute cosine similarity against all 2,957 chunks
3. Walk chunks in score order, collecting up to 2 chunks per resume, stopping at top-N resumes
4. No model inference — extract a clean snippet directly from the best chunk
5. Find first capital letter (max offset 80 chars) to avoid mid-sentence starts
6. Trim to last sentence/comma boundary within 250 characters

Multi-mode returns results in ~46ms for 5 candidates. Running model inference per candidate would take 7+ minutes on the available GPU slice.

---

## 10. API Server (`app.py`)

Built with FastAPI. Runs on port 8002.

### Endpoints

**GET `/health`** — Returns model status, device, and resume count.

**GET `/resumes`** — Lists all 440 indexed resumes with filename, student folder, and resume ID.

**POST `/ask`** — Main inference endpoint.

Request:
```json
{
  "question": "What programming languages does this candidate know?",
  "top_k": 5,
  "mode": "single"
}
```

Response:
```json
{
  "status": "FOUND",
  "answer": "Python, Java, and JavaScript",
  "sources": ["resume.pdf"],
  "context_used": "...",
  "candidates": []
}
```

### Inference Pipeline (Single Mode)

1. Retrieve top-K chunks via cosine similarity
2. Build prompt using exact SFT v2 format
3. Greedy decode up to 128 new tokens
4. Apply repetition penalty (1.5) on the last 80 tokens
5. Block special tokens from being generated
6. Stop at EOS token
7. Parse FOUND/NOT_FOUND prefix from output
8. Clean output with regex (remove repeated phrases)

### Greedy Decoding

The model uses argmax decoding — at each step, the token with the highest logit is selected. No sampling, no temperature, no top-p. This produces deterministic, clean output. The notebook experiments confirmed that sampling introduced noise; argmax gave the cleanest answers.

### Repetition Penalty

To prevent the model from repeating phrases, a penalty of 1.5 is applied to tokens that appeared in the last 80 generated tokens. Positive logits are divided by 1.5; negative logits are multiplied by 1.5. This makes repeated tokens less likely without completely blocking them.

### BFloat16 Inference

All forward passes use `torch.autocast` with `bfloat16`. This halves memory usage and speeds up inference on B200 GPUs with no meaningful quality loss.

---

## 11. Frontend (`resume-ui/`)

Built with React. Runs on port 3001.

### Features

- **Single mode**: Ask a question about the best-matching resume. Full model generation with FOUND/NOT_FOUND badges.
- **Multi-candidate mode**: Search across all 440 resumes. Returns ranked candidate cards with match scores, answer snippets, and expandable context.
- **Auto-detect**: Queries starting with "find", "who", "list", "show" are automatically routed to multi-candidate mode.
- **Dark theme**: Background `#0a0a0f`, accent purple `#7c6af7`.
- **Typing animation**: Three bouncing dots while waiting for response.
- **Score bars**: Visual match percentage for each candidate.
- **Context toggle**: Expand/collapse the raw retrieved chunks for any result.
- **Suggested questions**: Pre-built queries in the sidebar for both modes.

### Key Components

**CandidateCard**: Renders a single candidate result with rank badge, filename, student folder, score bar, answer snippet, and context toggle.

**Message**: Renders a chat message. Handles both single-mode (bubble with FOUND/NOT_FOUND badge) and multi-mode (list of CandidateCards).

**StatusBar**: Shows API connection status and resume count. Polls `/health` on load.

**isMultiQuery**: Regex function that auto-detects multi-candidate queries:
```javascript
/^(find|list|show|give me|who|which candidates?|search)/.test(lower) ||
lower.includes('candidates with') || lower.includes('who knows') || ...
```

---

## 12. Performance Numbers

| Metric | Value |
|---|---|
| Pretraining data | 1,000,000,000 tokens |
| Pretraining best val loss | 3.245 |
| Pretraining perplexity | ~25.6 |
| SFT v2 best selection score | 0.5175 |
| SFT v2 best real val loss | 0.5175 |
| Resumes indexed | 440 |
| Total chunks | 2,957 |
| Embedding dimensions | 384 |
| Multi-mode response time | ~46ms |
| Single-mode response time | ~1–3 seconds |
| Model parameters | 201,000,000 |
| GPU memory (inference) | ~26.3 GiB |
| Training hardware | NVIDIA B200 (8× available) |

---

## 13. What Makes This Different

Every other team fine-tuned an existing model. Fine-tuning is a valid technique. But it means the model's knowledge of language, grammar, reasoning, and the world came from someone else's training run — OpenAI, Meta, Alibaba. The team just adjusted the last few layers.

We started from random weights. The model did not know what a word was. It did not know what a sentence was. It did not know English existed. We gave it 1 billion tokens of text and let gradient descent teach it everything.

This means:

**We understand every parameter.** We wrote the attention mechanism. We wrote the RoPE implementation. We wrote the SwiGLU FFN. We wrote the training loop, the learning rate schedule, the gradient clipping, the checkpoint logic. There is no black box.

**We made real architectural decisions.** Grouped Query Attention with 14Q/7KV heads was a deliberate choice to reduce KV cache memory. RMSNorm over LayerNorm was a deliberate choice for speed. SwiGLU over ReLU was a deliberate choice for expressiveness. These are not defaults we accepted — they are decisions we made and understood.

**We ran a real training pipeline.** Pilot → 100M → 1B → SFT v1 → SFT v2 → SFT v3. Each stage built on the last. We tracked loss curves, identified overfitting at SFT v2 epoch 3, and selected the best checkpoint. This is how real LLM development works.

**We solved real engineering problems.** The multi-candidate mode was taking 7+ minutes because we were running model inference per candidate. We identified the bottleneck, switched to pure RAG embedding similarity, and brought it down to 46ms. Chunks were starting mid-sentence because of fixed-size character splitting. We rewrote the chunker to use sentence boundaries. These are not tutorial problems — they are production problems.

The model is not GPT-4. It is a 201M parameter model trained on 1B tokens. But it is ours, completely, from the first random weight to the last gradient update.

---

## Appendix: Training Chain Summary

```
Random weights (201M params)
        ↓
Pilot pretraining — 6.5M tokens — val loss 6.33
        ↓
100M token pretraining — val loss 4.36
        ↓
1B token pretraining — val loss 3.245  ← BASE MODEL
        ↓
SFT v1 — 5 epochs — val loss 0.715
        ↓
SFT v2 — 3 epochs — best val loss 0.5175 (epoch 2)  ← PRODUCTION
        ↓
SFT v3 — epoch 0 only (incomplete)
```

## Appendix: File Structure

```
~/twilight/
├── app.py                          # FastAPI server
├── build_index.py                  # RAG index builder
├── full_resume_rag_index.npz       # 440 resumes, 2957 chunks, 11.1MB
├── Downloaded_Resumes/             # 445 student PDF resumes
├── scratch_resume_lm_pilot/        # Pilot checkpoint + tokenizer
├── scratch_resume_lm_100m/         # 100M pretraining checkpoint
├── scratch_resume_lm_1b/           # 1B pretraining checkpoint
├── scratch_resume_lm_resume_sft/   # SFT v1 checkpoint
├── scratch_resume_lm_resume_sft_v2/# SFT v2 checkpoint (PRODUCTION)
├── scratch_resume_lm_resume_sft_v3/# SFT v3 checkpoint (incomplete)
├── scratch_resume_sft_v2_data/     # SFT v2 training data
├── scratch_resume_sft_v3_data/     # SFT v3 training data
└── resume-ui/                      # React frontend
    └── src/
        ├── App.js                  # Main React component
        └── App.css                 # Dark theme styles
```

---

*Built by Student15 — SECE 2026*
*Hardware: NVIDIA DGX B200 Server (8× B200 GPUs, 192.168.4.99)*
*Stack: PyTorch, FastAPI, React, sentence-transformers, PyMuPDF*
