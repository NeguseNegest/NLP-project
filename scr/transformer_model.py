import heapq
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

TRANSFORMER_DIR = Path(__file__).resolve().parent / "transformer"
sys.path.insert(0, str(TRANSFORMER_DIR))

from self_attention import MultiHeadSelfAttention
from tokenizer import TinyStoriesTokenizer


@dataclass
class Config:
    vocab_size: int = 5000
    number_of_transformer_blocks: int = 4
    number_of_attention_heads: int = 4
    vector_dim: int = 256
    block_size: int = 512
    dropout_prob: float = 0.1
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 1e-6
    no_of_epochs: int = 1


class PositionwiseFFN(nn.Module):
    def __init__(self, vector_dim, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(vector_dim, 4 * vector_dim, bias=True)
        self.fc2 = nn.Linear(4 * vector_dim, vector_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, vector_dim, n_heads, block_size, dropout_prob):
        super().__init__()
        self.attn = MultiHeadSelfAttention(vector_dim, n_heads, block_size, is_causal=True)
        self.ffn = PositionwiseFFN(vector_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TinyStoriesLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.vector_dim)
        self.positional = nn.Parameter(torch.randn(1, config.block_size, config.vector_dim))
        self.transformers = nn.ModuleList([
            Block(config.vector_dim, config.number_of_attention_heads,
                  config.block_size, config.dropout_prob)
            for _ in range(config.number_of_transformer_blocks)
        ])
        self.final = nn.Linear(config.vector_dim, config.vocab_size)

    def forward(self, x):
        _, T = x.shape
        h = self.embed(x) + self.positional[:, :T, :]
        for block in self.transformers:
            h = block(h)
        return self.final(h)

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = Config(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Model loaded from {checkpoint_path} "
              f"(epoch {checkpoint['epoch']}, iter {checkpoint['iteration']})")
        return model


class TransformerPredictor:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = device
        self.tokenizer = TinyStoriesTokenizer.load(tokenizer_path)
        self.model = TinyStoriesLM.load(checkpoint_path, device=device)
        self.model.eval()

        self.word_vocab = []
        self.word_counts = {}
        self.prefix_index = {}

    def load_word_vocab_from_ngram(self, ngram_model_path):
        with open(ngram_model_path, "rb") as f:
            ngram = pickle.load(f)
        special = {ngram.start_token, ngram.end_token, ngram.unk_token}
        self.word_vocab = [w for w in ngram.vocab if w not in special]
        self.word_counts = dict(ngram.word_counts)
        self._build_prefix_index()
        print(f"Loaded {len(self.word_vocab):,} words from n-gram model.")

    def build_word_vocab_from_text(self, text_path, min_count=2):
        from collections import Counter
        counts = Counter()
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.lower().split():
                    w = w.strip(".,!?;:\"'()[]{}").strip()
                    if w:
                        counts[w] += 1
        self.word_vocab = [w for w, c in counts.items() if c >= min_count]
        self.word_counts = dict(counts)
        self._build_prefix_index()
        print(f"Built vocab of {len(self.word_vocab):,} words from {text_path}.")

    def _build_prefix_index(self):
        self.prefix_index = {"": list(self.word_vocab)}
        for word in self.word_vocab:
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if prefix not in self.prefix_index:
                    self.prefix_index[prefix] = []
                self.prefix_index[prefix].append(word)

    def parse_text_input(self, text):
        if not text:
            return [], ""
        lower = text.lower()
        if lower.endswith(" "):
            return lower.strip().split(), ""
        words = lower.strip().split()
        if not words:
            return [], ""
        return words[:-1], words[-1]

    def get_candidates(self, prefix=""):
        return self.prefix_index.get(prefix.lower().strip(), [])

    def predict(self, context, prefix="", top_k=5, include_scores=True):
        candidates = self.get_candidates(prefix)
        if not candidates:
            return []

        scores = self._score_candidates(context, candidates)

        ranked = heapq.nsmallest(
            top_k,
            zip(candidates, scores),
            key=lambda item: (
                -item[1],
                -self.word_counts.get(item[0], 0),
                len(item[0]),
                item[0],
            ),
        )

        return list(ranked) if include_scores else [w for w, _ in ranked]

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        return self.predict(context, prefix, top_k, include_scores)

    def _score_candidates(self, context, candidates):
        # Score each candidate by the probability of its first BPE token given the context.
        # This is an approximation — two words sharing the same first token get the same score,
        # but in practice the prefix filter makes collisions rare.
        context_text = " ".join(context)

        if not context_text:
            freq_total = sum(self.word_counts.get(w, 1) for w in candidates)
            return [self.word_counts.get(w, 1) / freq_total for w in candidates]

        _, context_ids = self.tokenizer.tokenize(context_text)
        if not context_ids:
            return [1.0 / len(candidates)] * len(candidates)

        context_ids = context_ids[-(self.model.config.block_size - 1):]
        x = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)

        next_probs = torch.softmax(logits[0, -1, :], dim=-1).cpu()
        vocab_size = len(next_probs)

        scores = []
        for word in candidates:
            _, word_ids = self.tokenizer.tokenize(" " + word)
            if not word_ids or word_ids[0] >= vocab_size:
                scores.append(0.0)
            else:
                scores.append(next_probs[word_ids[0]].item())

        return scores
