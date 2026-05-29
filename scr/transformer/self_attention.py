import math
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, vector_dim, block_size, is_causal=False):
        super().__init__()
        self.vector_dim = vector_dim
        self.is_causal = is_causal
        self.wq = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wk = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wv = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)
        if self.is_causal:
            causal_mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0)
            self.register_buffer("mask", causal_mask)

    def compute_attention(self, q, k, v):
        if hasattr(F, "scaled_dot_product_attention"):
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=self.is_causal,
                    scale=1.0 / math.sqrt(self.vector_dim),
                )
            except TypeError:
                pass

        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.vector_dim)
        if self.is_causal and hasattr(self, "mask"):
            t = q.size(-2)
            mask = self.mask[:, :t, :t]
            if q.dim() == 4:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        return torch.softmax(attn_scores, dim=-1) @ v

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        return self.wo(self.compute_attention(q, k, v))


class MultiHeadSelfAttention(SelfAttention):
    def __init__(self, vector_dim, n_heads, block_size, is_causal=False):
        super().__init__(vector_dim, block_size, is_causal)
        self.att_dim = vector_dim // n_heads
        self.n_heads = n_heads

    def reshape_for_multihead_attention(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.att_dim).transpose(1, 2)

    def reshape_after_multihead_attention(self, x):
        b, h, t, att_dim = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * att_dim)

    def forward(self, x):
        q = self.reshape_for_multihead_attention(self.wq(x))
        k = self.reshape_for_multihead_attention(self.wk(x))
        v = self.reshape_for_multihead_attention(self.wv(x))
        values = self.reshape_after_multihead_attention(self.compute_attention(q, k, v))
        return self.wo(values)
