import heapq
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


def resolve_torch_device(device="auto"):
    """Resolve auto/cuda/mps/cpu safely."""
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if requested.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS was requested, but MPS is not available. Falling back to CPU.")
        return torch.device("cpu")

    return requested

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

    def hidden(self, x):
        _, T = x.shape
        h = self.embed(x) + self.positional[:, :T, :]
        for block in self.transformers:
            h = block(h)
        return h

    def forward(self, x):
        return self.final(self.hidden(x))

    @classmethod
    def load(cls, checkpoint_path, device="auto"):
        device = resolve_torch_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = Config(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Model loaded from {checkpoint_path} "
              f"(epoch {checkpoint['epoch']}, iter {checkpoint['iteration']}) "
              f"on device {device}")
        return model


class TransformerPredictor:
    def __init__(self, checkpoint_path, tokenizer_path, device="auto"):
        self.device = resolve_torch_device(device)
        self.tokenizer = TinyStoriesTokenizer.load(tokenizer_path)
        self.model = TinyStoriesLM.load(checkpoint_path, device=self.device)
        self.model.eval()

        self.word_vocab = []
        self.word_vocab_set = set()
        self.word_counts = {}
        self.prefix_index = {}
        self._word_tokens = {}
        self._logits_cache_key = None
        self._logits_cache_probs = None
        self._full_score_cache_key = None
        self._full_rank_cache = None

    def load_word_vocab_from_ngram(self, ngram_model_path):
        ngram_dir = str(Path(ngram_model_path).resolve().parents[2] / "scr" / "ngram")
        if ngram_dir not in sys.path:
            sys.path.insert(0, ngram_dir)
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
        self.word_vocab_set = set(self.word_vocab)
        self.prefix_index = {"": list(self.word_vocab)}
        for word in self.word_vocab:
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if prefix not in self.prefix_index:
                    self.prefix_index[prefix] = []
                self.prefix_index[prefix].append(word)
        self._word_tokens = {}
        for word in self.word_vocab:
            _, ids = self.tokenizer.tokenize(" " + word)
            self._word_tokens[word] = ids if ids else [0]
        self._full_score_cache_key = None
        self._full_rank_cache = None
        print(f"Pre-tokenized {len(self._word_tokens):,} words.")

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
        normalized_prefix = prefix.lower().strip()
        candidates = self.get_candidates(normalized_prefix)
        if not candidates:
            return []

        context_state = self._context_state(context)
        use_full_rank_cache = (
            context_state[0] == "model"
            and self._full_score_cache_key == context_state
            and self._full_rank_cache is not None
        )

        if use_full_rank_cache:
            ranked = self._top_from_full_rank_cache(normalized_prefix, top_k)
        else:
            scores = self._score_candidates(context, candidates, context_state=context_state)
            scored_candidates = list(zip(candidates, scores))

            if (
                context_state[0] == "model"
                and normalized_prefix == ""
                and len(candidates) == len(self.word_vocab)
            ):
                self._full_score_cache_key = context_state
                self._full_rank_cache = sorted(scored_candidates, key=self._rank_key)
                ranked = self._full_rank_cache[:top_k]
            else:
                ranked = self._rank_items(scored_candidates, top_k)

        return list(ranked) if include_scores else [w for w, _ in ranked]

    def _rank_key(self, item):
        return (
            -item[1],
            -self.word_counts.get(item[0], 0),
            len(item[0]),
            item[0],
        )

    def _rank_items(self, items, top_k):
        return heapq.nsmallest(
            top_k,
            items,
            key=self._rank_key,
        )

    def _top_from_full_rank_cache(self, prefix, top_k):
        if not prefix:
            return self._full_rank_cache[:top_k]

        ranked = []
        for word, score in self._full_rank_cache:
            if word.startswith(prefix):
                ranked.append((word, score))
                if len(ranked) >= top_k:
                    break
        return ranked

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        return self.predict(context, prefix, top_k, include_scores)

    def _context_state(self, context):
        context_text = " ".join(context)

        if not context_text:
            return ("frequency", ())

        _, context_ids = self.tokenizer.tokenize(" " + context_text)
        if not context_ids:
            return ("uniform", ())

        context_ids = tuple(context_ids[-(self.model.config.block_size - 1):])
        return ("model", context_ids)

    def _score_candidates(self, context, candidates, context_state=None):
        context_state = context_state or self._context_state(context)
        context_mode, context_ids = context_state

        if context_mode == "frequency":
            freq_total = sum(self.word_counts.get(w, 1) for w in candidates)
            return [self.word_counts.get(w, 1) / freq_total for w in candidates]

        if context_mode == "uniform":
            return [1.0 / len(candidates)] * len(candidates)

        context_ids = list(context_ids)
        ctx_len = len(context_ids)
        block_size = self.model.config.block_size
        cache_key = tuple(context_ids)

        word_toks_list = [self._word_tokens.get(w, [0]) for w in candidates]

        single_indices = [i for i, toks in enumerate(word_toks_list) if len(toks) == 1]
        multi_indices  = [i for i, toks in enumerate(word_toks_list) if len(toks) > 1]

        all_scores = [0.0] * len(candidates)

        if single_indices:
            if cache_key != self._logits_cache_key:
                x = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    h = self.model.hidden(x)
                    logits = self.model.final(h[:, -1, :])
                self._logits_cache_probs = torch.softmax(logits[0], dim=-1).detach().cpu()
                self._logits_cache_key = cache_key
            probs = self._logits_cache_probs
            vocab_size = len(probs)
            for i in single_indices:
                tid = word_toks_list[i][0]
                all_scores[i] = probs[tid].item() if tid < vocab_size else 0.0

        if multi_indices:
            chunk_size = 8192 if self.device.type == "cuda" else 1024
            for chunk_start in range(0, len(multi_indices), chunk_size):
                chunk_idx = multi_indices[chunk_start:chunk_start + chunk_size]
                chunk_toks = [word_toks_list[i] for i in chunk_idx]
                max_word_len = max(len(toks) for toks in chunk_toks)
                target_len = min(ctx_len + max_word_len - 1, block_size)

                batch = []
                for toks in chunk_toks:
                    inp = (context_ids + list(toks[:-1]))[-block_size:]
                    inp = inp + [0] * max(0, target_len - len(inp))
                    batch.append(inp[:target_len])

                x = torch.tensor(batch, dtype=torch.long, device=self.device)
                row_ids = []
                pos_ids = []
                token_ids = []
                candidate_ids = []
                for k, toks in enumerate(chunk_toks):
                    for j, tok_id in enumerate(toks):
                        row_ids.append(k)
                        pos_ids.append(min(ctx_len - 1 + j, target_len - 1))
                        token_ids.append(tok_id)
                        candidate_ids.append(k)

                with torch.inference_mode():
                    h = self.model.hidden(x)
                    row_t = torch.tensor(row_ids, dtype=torch.long, device=self.device)
                    pos_t = torch.tensor(pos_ids, dtype=torch.long, device=self.device)
                    tok_t = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                    cand_t = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)

                    selected_h = h[row_t, pos_t, :]
                    selected_logits = self.model.final(selected_h)
                    vocab_size = selected_logits.shape[-1]
                    safe_tok_t = tok_t.clamp(0, vocab_size - 1)
                    token_log_p = torch.log_softmax(selected_logits, dim=-1)[
                        torch.arange(safe_tok_t.numel(), device=self.device), safe_tok_t
                    ]
                    invalid = (tok_t < 0) | (tok_t >= vocab_size)
                    token_log_p = torch.where(
                        invalid,
                        torch.full_like(token_log_p, -20.0),
                        token_log_p,
                    )

                    score_sums = torch.zeros(len(chunk_idx), dtype=token_log_p.dtype, device=self.device)
                    score_sums.index_add_(0, cand_t, token_log_p)
                    score_probs = torch.exp(score_sums).detach().cpu().tolist()

                for orig_i, score in zip(chunk_idx, score_probs):
                    all_scores[orig_i] = score

        return all_scores
