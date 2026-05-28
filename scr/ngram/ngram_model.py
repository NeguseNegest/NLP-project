import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import heapq

class NGramModel:

    def __init__(self, max_n_gram=4, min_count=1, unk_token="<UNK>", start_token="<s>", end_token="</s>"):
        self.max_n_gram = max_n_gram
        self.min_count = min_count
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token

        self.vocab = set()
        self.word_counts = Counter()
        self.total_words = 0

        self.counts = {n: defaultdict(Counter) for n in range(1, self.max_n_gram + 1)}

        self.prefix_index = {}
        self.non_special_vocab = []

        self.context_totals = {n: {} for n in range(1, self.max_n_gram + 1)}

        self.is_trained = False



    def train(self, tokenized_sentences, show_progress=True):
    
        sentences = [list(sentence) for sentence in tokenized_sentences]

        if not sentences:
            raise ValueError("Cannot train on an empty dataset.")

        raw_counts = Counter()

        sentence_iterator = tqdm(sentences, desc="Building vocabulary", disable=not show_progress)

        for sentence in sentence_iterator:
            cleaned_sentence = [word.lower().strip() for word in sentence if word.strip()]
            raw_counts.update(cleaned_sentence)

        self.vocab = {word for word, count in raw_counts.items() if count >= self.min_count}

        self.vocab.add(self.unk_token)
        self.vocab.add(self.end_token)

        self.word_counts = Counter()
        self.counts = {n: defaultdict(Counter) for n in range(1, self.max_n_gram + 1)}

        sentence_iterator = tqdm(sentences, desc="Training n-gram model", disable=not show_progress)

        for sentence in sentence_iterator:
            normalized_sentence = [self._normalize_word(word) for word in sentence if word.strip()]
            self._update_counts_from_sentence(normalized_sentence)

        self.total_words = sum(self.word_counts.values())
        self._build_prefix_index()
        self._build_context_totals()
        self.is_trained = True


    def _update_counts_from_sentence(self, sentence):
        padded = [self.start_token] * (self.max_n_gram - 1) + list(sentence) + [self.end_token]

        for word in list(sentence) + [self.end_token]:
            self.word_counts[word] += 1

        for i in range(self.max_n_gram - 1, len(padded)):
            focus_word = padded[i]

            for n in range(1, self.max_n_gram + 1):
                context_size = n - 1

                if context_size == 0:
                    context = tuple()
                else:
                    context = tuple(padded[i - context_size:i])

                self.counts[n][context][focus_word] += 1



    def probability(self, context, word, n=4):
        self._check_trained()
        self._check_n(n)

        word = self._normalize_candidate_word(word)
        context_tuple = self._get_context_tuple(context, n)

        vocab_size = len(self.vocab)

        context_counter = self.counts[n].get(context_tuple)
        count = 0 if context_counter is None else context_counter.get(word, 0)

        context_total = self.context_totals[n].get(context_tuple, 0)

        numerator = count + 1
        denominator = context_total + vocab_size

        return numerator / denominator


    def log_probability(self, context, word, n=4):
        return math.log(self.probability(context, word, n=n))


    def interpolated_probability(self, context, word, lambdas=None):
        self._check_trained()

        if lambdas is None:
            lambdas = {n: 1.0 / self.max_n_gram for n in range(1, self.max_n_gram + 1)}

        self._validate_lambdas(lambdas)

        probability_sum = 0.0

        for n, weight in lambdas.items():
            probability_sum += weight * self.probability(context, word, n=n)

        return probability_sum

    def interpolated_log_probability(self, context, word, lambdas=None):
        return math.log(self.interpolated_probability(context, word, lambdas=lambdas))

    def predict(self, context, prefix="", n=4, top_k=5, include_scores=True):
        self._check_trained()
        self._check_n(n)

        candidates = self.get_candidates(prefix)

        scored_candidates = [(word, self.probability(context, word, n=n)) for word in candidates]
        ranked = self._rank_candidates(scored_candidates, top_k=top_k)

        if include_scores:
            return ranked

        return [word for word, _ in ranked]

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        self._check_trained()

        candidates = self.get_candidates(prefix)

        scored_candidates = [(word, self.interpolated_probability(context, word, lambdas=lambdas)) for word in candidates]

        ranked = self._rank_candidates(scored_candidates, top_k=top_k)

        if include_scores:
            return ranked

        return [word for word, _ in ranked]

    def get_candidates(self, prefix=""):
        self._check_trained()

        prefix = prefix.lower().strip()

        return self.prefix_index.get(prefix, [])
    

    def _build_context_totals(self):
        self.context_totals = {}

        for n in range(1, self.max_n_gram + 1):
            self.context_totals[n] = {}

            for context, counter in self.counts[n].items():
                self.context_totals[n][context] = sum(counter.values())

    def next_word_distribution(self, context, n=4):
        self._check_trained()
        self._check_n(n)

        distribution = {}

        for word in self.vocab:
            if word in {self.start_token, self.unk_token}:
                continue

            distribution[word] = self.probability(context, word, n=n)

        return distribution

    def parse_text_input(self, text):
        if text is None:
            return [], ""

        lower_text = text.lower()

        if lower_text.endswith(" "):
            words = lower_text.strip().split()
            return words, ""

        words = lower_text.strip().split()

        if not words:
            return [], ""

        context = words[:-1]
        prefix = words[-1]

        return context, prefix

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        path = Path(path)

        with path.open("rb") as f:
            model = pickle.load(f)

        return model

    def _normalize_word(self, word):
        word = word.lower().strip()

        if word in self.vocab:
            return word

        return self.unk_token

    def _normalize_candidate_word(self, word):
        word = word.lower().strip()

        if word in self.vocab:
            return word

        return self.unk_token

    def _get_context_tuple(self, context, n):
        context_size = n - 1

        if context_size == 0:
            return tuple()

        normalized_context = []

        for word in context:
            word = word.lower().strip()

            if word in self.vocab:
                normalized_context.append(word)
            else:
                normalized_context.append(self.unk_token)

        if len(normalized_context) < context_size:
            missing = context_size - len(normalized_context)
            normalized_context = [self.start_token] * missing + normalized_context

        return tuple(normalized_context[-context_size:])

    def _build_prefix_index(self):
        special_tokens = {self.start_token, self.end_token, self.unk_token}

        self.non_special_vocab = [word for word in self.vocab if word not in special_tokens]
        self.prefix_index = {"": self.non_special_vocab}

        for word in self.non_special_vocab:
            for i in range(1, len(word) + 1):
                prefix = word[:i]

                if prefix not in self.prefix_index:
                    self.prefix_index[prefix] = []

                self.prefix_index[prefix].append(word)

    def _rank_candidates(self, scored_candidates, top_k):
        if top_k <= 0:
            return []

        return heapq.nsmallest(
            top_k,
            scored_candidates,
            key=lambda item: (
                -item[1],
                -self.word_counts.get(item[0], 0),
                len(item[0]),
                item[0],
            ),
        )

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("NGramModel must be trained before it can be used.")

    def _check_n(self, n):
        if n < 1 or n > self.max_n_gram:
            raise ValueError(f"n must be between 1 and {self.max_n_gram}, got n={n}.")

    def _validate_lambdas(self, lambdas):
        if not lambdas:
            raise ValueError("lambdas cannot be empty.")

        for n, weight in lambdas.items():
            self._check_n(n)

            if weight < 0:
                raise ValueError("Interpolation weights must be non negative.")

        total = sum(lambdas.values())

        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"interpolation weights must sum to 1. Got {total}.")

    

    
    def __repr__(self):
        return f"NGramModel(max_n_gram={self.max_n_gram}, min_count={self.min_count}, vocab_size={len(self.vocab)}, is_trained={self.is_trained})"
