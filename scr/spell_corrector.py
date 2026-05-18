import heapq
import math
import random


_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_KEY_POS = {}
for _row_idx, _row in enumerate(_KEYBOARD_ROWS):
    for _col_idx, _ch in enumerate(_row):
        _KEY_POS[_ch] = (_row_idx, _col_idx + _row_idx * 0.5)


def _keyboard_sub_cost(c1, c2):
    if c1 == c2:
        return 0.0
    if c1 not in _KEY_POS or c2 not in _KEY_POS:
        return 1.0
    r1, x1 = _KEY_POS[c1]
    r2, x2 = _KEY_POS[c2]
    dist = math.sqrt((r1 - r2) ** 2 + (x1 - x2) ** 2)
    return 0.5 if dist <= 1.5 else 1.0


def damerau_levenshtein(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = float(i)
    for j in range(len2 + 1):
        dp[0][j] = float(j)
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[len1][len2]


def weighted_damerau_levenshtein(s1, s2, w_insert=1.0, w_delete=1.0,
                                  w_transpose=0.5, use_keyboard=False):
    len1, len2 = len(s1), len(s2)
    dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i * w_delete
    for j in range(len2 + 1):
        dp[0][j] = j * w_insert
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                sub_cost = 0.0
            elif use_keyboard:
                sub_cost = _keyboard_sub_cost(s1[i - 1], s2[j - 1])
            else:
                sub_cost = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + w_delete,
                dp[i][j - 1] + w_insert,
                dp[i - 1][j - 1] + sub_cost,
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + w_transpose)
    return dp[len1][len2]


def introduce_typo(word, max_edit_dist=1, seed=None):
    if seed is not None:
        random.seed(seed)
    if len(word) < 2:
        return word

    ops = ["delete", "transpose", "substitute", "insert"]
    n_ops = random.randint(1, max_edit_dist)
    result = list(word)

    for _ in range(n_ops):
        if not result:
            break
        op = random.choice(ops)
        i = random.randint(0, len(result) - 1)

        if op == "delete":
            result.pop(i)
        elif op == "substitute":
            result[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        elif op == "insert":
            result.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == "transpose" and i < len(result) - 1:
            result[i], result[i + 1] = result[i + 1], result[i]

    return "".join(result)


class SpellCorrector:

    def __init__(self, predictor, max_edit_dist=2, lambda_edit=0.5, mu_freq=0.1,
                 weight_mode="uniform"):
        self.predictor = predictor
        self.max_edit_dist = max_edit_dist
        self.lambda_edit = lambda_edit
        self.mu_freq = mu_freq
        self.weight_mode = weight_mode

        self._vocab = self._get_word_vocab()
        self._counts = predictor.word_counts if hasattr(predictor, "word_counts") else {}
        self._max_freq_log = math.log1p(max(self._counts.values())) if self._counts else 1.0

    def predict(self, context, prefix="", top_k=5, include_scores=True, use_lm=True):
        if not prefix:
            return self._fallback_predict(context, prefix, top_k, include_scores)

        spell_candidates = self.get_spell_candidates(prefix)
        if not spell_candidates:
            return []

        words = [w for w, _ in spell_candidates]
        edit_dists = {w: d for w, d in spell_candidates}

        scored = []
        if use_lm:
            lm_scores = self._score_words(context, words)
            for word, lm_score in zip(words, lm_scores):
                log_lm = math.log(lm_score + 1e-12)
                edit_pen = edit_dists[word] / (self.max_edit_dist + 1e-9)
                norm_freq = math.log1p(self._counts.get(word, 1)) / self._max_freq_log
                combined = log_lm - self.lambda_edit * edit_pen + self.mu_freq * norm_freq
                scored.append((word, combined))
        else:
            for word in words:
                edit_pen = edit_dists[word] / (self.max_edit_dist + 1e-9)
                norm_freq = math.log1p(self._counts.get(word, 1)) / self._max_freq_log
                combined = -self.lambda_edit * edit_pen + self.mu_freq * norm_freq
                scored.append((word, combined))

        top = heapq.nsmallest(top_k, scored, key=lambda x: -x[1])
        return top if include_scores else [w for w, _ in top]

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        return self.predict(context, prefix, top_k, include_scores)

    def parse_text_input(self, text):
        return self.predictor.parse_text_input(text)

    def get_spell_candidates(self, prefix):
        prefix = prefix.lower().strip()
        p_len = len(prefix)
        if p_len == 0:
            return [(w, 0) for w in self._vocab]

        candidates = []
        for word in self._vocab:
            if len(word) < p_len - self.max_edit_dist:
                continue
            if self.weight_mode == "operation":
                dist = weighted_damerau_levenshtein(prefix, word[:p_len],
                                                    w_transpose=0.5, use_keyboard=False)
            elif self.weight_mode == "keyboard":
                dist = weighted_damerau_levenshtein(prefix, word[:p_len],
                                                    w_transpose=0.5, use_keyboard=True)
            else:
                dist = damerau_levenshtein(prefix, word[:p_len])
            if dist <= self.max_edit_dist:
                candidates.append((word, dist))

        return candidates

    @property
    def word_vocab(self):
        return self._vocab

    @property
    def word_vocab_set(self):
        return set(self._vocab)

    def _get_word_vocab(self):
        if hasattr(self.predictor, "word_vocab"):
            return self.predictor.word_vocab
        if hasattr(self.predictor, "non_special_vocab"):
            return self.predictor.non_special_vocab
        return []

    def _score_words(self, context, words):
        if hasattr(self.predictor, "_score_candidates"):
            return self.predictor._score_candidates(context, words)
        return [self.predictor.interpolated_probability(context, w) for w in words]

    def _fallback_predict(self, context, prefix, top_k, include_scores):
        if hasattr(self.predictor, "predict_interpolated"):
            return self.predictor.predict_interpolated(
                context, prefix, top_k, include_scores=include_scores
            )
        return self.predictor.predict(context, prefix, top_k, include_scores)
