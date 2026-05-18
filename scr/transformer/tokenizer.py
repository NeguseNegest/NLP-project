import re
import json
from collections import Counter, defaultdict


class TinyStoriesTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = []
        self.ids = {}

    def _pretokenize(self, text):
        return re.findall(r" ?\w[\w']*| ?[^\w\s]", text)

    def _generate_word_frequencies(self, words):
        return dict(Counter(tuple(word) for word in words))

    def _count_token_bigrams(self, word_freqs):
        bigram_counts = defaultdict(int)
        for chars, count in word_freqs.items():
            for bigram in zip(chars, chars[1:]):
                bigram_counts[bigram] += count
        return bigram_counts

    def _find_most_frequent_token_bigram(self, word_freqs):
        bigram_counts = self._count_token_bigrams(word_freqs)
        return max(bigram_counts, key=lambda b: bigram_counts[b])

    def _merge_bigram(self, word_freqs, best_bigram, new_token):
        left, right = best_bigram
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            merged = []
            i = 0
            while i < len(word_tuple):
                if word_tuple[i] == left and i + 1 < len(word_tuple) and word_tuple[i + 1] == right:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(merged)] = freq
        return new_word_freqs

    def train(self, corpus_path, max_lines=None):
        lines = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))

        words = self._pretokenize("\n".join(lines))
        word_freqs = self._generate_word_frequencies(words)

        unique_chars = set(char for word_tuple in word_freqs for char in word_tuple)
        self.vocab = ["<UNK>"] + list(unique_chars)

        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            best_bigram = self._find_most_frequent_token_bigram(word_freqs)
            self.merges[best_bigram] = i
            new_token = "".join(best_bigram)
            self.vocab.append(new_token)
            word_freqs = self._merge_bigram(word_freqs, best_bigram, new_token)
            if (i + 1) % 500 == 0:
                print(f"  Merge {i+1}/{num_merges}: {''.join(best_bigram)}")

        self.ids = {v: k for k, v in enumerate(self.vocab)}
        print(f"Tokenizer trained. Vocab size: {len(self.vocab)}")

    def tokenize(self, text):
        tokens = []
        for word in self._pretokenize(text):
            parts = list(word)
            while len(parts) > 1:
                bigrams = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
                eligible = [(i, b) for i, b in enumerate(bigrams) if b in self.merges]
                if not eligible:
                    break
                i, best = min(eligible, key=lambda x: self.merges[x[1]])
                parts = parts[:i] + ["".join(best)] + parts[i + 2:]
            tokens.extend(parts)
        return tokens, [self.ids.get(t, 0) for t in tokens]

    def save(self, path):
        serializable_merges = {f"{k[0]}<SPLIT>{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab": self.vocab,
            "merges": serializable_merges,
            "vocab_size": self.vocab_size,
            "ids": self.ids,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab_size=data["vocab_size"])
        instance.vocab = data["vocab"]
        instance.ids = data["ids"]
        instance.merges = {tuple(k.split("<SPLIT>")): v for k, v in data["merges"].items()}
        return instance
