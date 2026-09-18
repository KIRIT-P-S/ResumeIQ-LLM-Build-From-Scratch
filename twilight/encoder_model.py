"""
Encoder (BERT-style) trained from scratch for JD-Resume matching.
Bidirectional self-attention — no causal mask.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int = 32000
    dim: int = 384
    layers: int = 6
    heads: int = 6
    hidden: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_id: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class EncoderAttention(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.h = c.heads
        self.d = c.dim // c.heads
        self.q = nn.Linear(c.dim, c.dim, bias=False)
        self.k = nn.Linear(c.dim, c.dim, bias=False)
        self.v = nn.Linear(c.dim, c.dim, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d).transpose(1, 2)
        # Bidirectional — no causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o(out)


class EncoderBlock(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.n1 = RMSNorm(c.dim)
        self.n2 = RMSNorm(c.dim)
        self.attn = EncoderAttention(c)
        self.gate = nn.Linear(c.dim, c.hidden, bias=False)
        self.up   = nn.Linear(c.dim, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.dim, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.n1(x), mask))
        y = self.n2(x)
        return x + self.drop(self.down(F.silu(self.gate(y)) * self.up(y)))


class ScratchEncoder(nn.Module):
    """
    BERT-style encoder. Uses [CLS] token (position 0) as sentence embedding.
    Total params ~22M with default config (dim=384, layers=6).
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_id)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.dim)
        self.drop      = nn.Dropout(config.dropout)
        self.blocks    = nn.ModuleList([EncoderBlock(config) for _ in range(config.layers)])
        self.norm      = RMSNorm(config.dim)
        # Projection head for contrastive similarity (maps to 128-dim unit sphere)
        self.proj      = nn.Linear(config.dim, 128, bias=False)

    def forward(self, ids, mask=None):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        cls = x[:, 0]           # [CLS] token representation
        return F.normalize(self.proj(cls), dim=-1)   # L2-normalized embedding

    def encode_mean(self, ids, mask=None):
        """Mean pooling over non-padding tokens — alternative to CLS."""
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)
        return F.normalize(self.proj(pooled), dim=-1)
