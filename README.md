# NLP-project

![Screenshot](miscl/image_web.png)

Project folder overivew 

```
NLP-Project/

├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
│
├── models/
│   ├── ngram/
│   └── transformer/
│
├── results/
│   ├── metrics/
│   └── plots/
│
├── src/
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── ngram_model.py
│   ├── transformer_model.py
│   ├── spell_corrector.py
│   ├── suggestion_engine.py
│   ├── evaluate.py
│   └── gui_app.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Do the following to test the word predictor, run first the folllowing command:

```bash
 python scr/ngram/ngram_train.py 
 ``` 

 ## The run the gui and start typing

 ```bash 
 python scr/gui_app.py
 ```



 # Results

 ## Summary of Language Model Evaluation

The best interpolation weights found during tuning were:

| N-gram model | Lambda weight |
|---|---:|
| Unigram (1-gram) | 0.0 |
| Bigram (2-gram) | 0.0 |
| Trigram (3-gram) | 0.1 |
| 4-gram | 0.9 |

This means the final model relies mostly on the **4-gram model**, with a small contribution from the **3-gram model**.

## Best Validation Result

The best validation performance was obtained with:

- **Top-k:** 3 suggestions
- **Saved keystrokes:** 122,158
- **Saved keystroke ratio:** 68.04%
- **Top-k accuracy:** 51.48%
- **Success rate:** 95.40%
- **Appeared rate:** 99.88%
- **Mean characters typed before suggestion:** 1.24

This means that when showing the top 3 suggestions, the correct word appeared in the suggestions about **51.5%** of the time. The system also saved about **68%** of the total characters that would otherwise have been typed.

## Test Results by Number of Suggestions

| Top-k | Top-k accuracy | Saved keystroke ratio | Saved keystrokes |
|---:|---:|---:|---:|
| 1 | 32.53% | 57.80% | 147,550 |
| 2 | 44.10% | 65.61% | 167,488 |
| 3 | 51.42% | 68.03% | 173,665 |
| 4 | 56.47% | 69.29% | 176,900 |

Increasing the number of suggestions improves both accuracy and saved keystrokes. However, the improvement becomes smaller after `top_k = 3`. For example, going from 3 to 4 suggestions only increases the saved keystroke ratio from **68.03%** to **69.29%**.

## Cross-Entropy and Perplexity

The model achieved:

- **Cross-entropy:** 5.55
- **Perplexity:** 258.52

The perplexity means that, on average, the model is about as uncertain as choosing between **258 possible next words**. A lower perplexity would indicate a better language model, but this value is reasonable for a simple word-level n-gram model.

## Conclusion

The best tuned model used interpolation weights of **λ₃ = 0.1** and **λ₄ = 0.9**, meaning it mainly depends on 4-word context. With `top_k = 3`, the model achieved a good balance between prediction quality and usability, reaching **51.42% top-k accuracy** and saving about **68.03%** of keystrokes on the test set. Although `top_k = 4` gives the highest accuracy and keystroke savings, `top_k = 3` is a reasonable tradeoff because it gives strong performance while showing fewer suggestions.

---

## Spell Correction

## What is it and why does it matter?

Normal word prediction assumes you type correctly — it just tries to finish what you started. But what if you make a typo? That's where spell correction comes in. Instead of only looking for words that *start with* what you typed, it also considers words that are *close to* what you typed, even if a character is wrong, swapped, or missing.

In the GUI, spell correction runs alongside word prediction automatically. The moment you enable it and start typing, the suggestion list already accounts for potential typos — so even if you type `"hte"` instead of `"the"`, the correct word can still appear in the suggestions.

## How does it work under the hood?

The core idea is **edit distance** — a measure of how many single-character operations it takes to turn what you typed into a real word. The allowed operations are insert, delete, substitute, or swap two adjacent characters (transposition). For example:

- `"hte"` → `"the"` costs 1 operation (swap `h` and `t`)
- `"recieve"` → `"receive"` costs 1 operation (swap `i` and `e`)
- `"wrold"` → `"world"` costs 1 operation (swap `r` and `l`)

Only words within the edit distance threshold are considered as candidates. Then each candidate gets a score.

With a language model:

```text
score(c) = log P(c | context)  −  λ · (dist / max_dist)  +  μ · log(freq(c) + 1) / max_log_freq
```

Without a language model:

```text
score(c) =  −  λ · (dist / max_dist)  +  μ · log(freq(c) + 1) / max_log_freq
```

Where:

- `P(c | context)` — probability the language model assigns to candidate `c` given the preceding words *(with-LM formula only)*
- `dist` — edit distance between what you typed and the candidate; **the distance function itself depends on the strategy**: S1/S2 use standard Damerau-Levenshtein (all ops cost 1.0), S3 uses a weighted variant where some operations cost less than 1.0
- `max_dist` — the normalizing threshold set by the strategy: `1` for S1, `2` for S2 and S3
- `freq(c)` — how often the word appears in training data
- `λ = 0.5` — how much to penalise candidates that are far from what you typed
- `μ = 0.1` — small frequency boost so common words don't get pushed out

The candidates are ranked by this score, and the top suggestions are shown.

## Choosing a Strategy (S1 / S2 / S3)

The strategy determines two things: the **maximum edit distance** and the **cost model** used when computing that distance.

---

### S1 — Standard Damerau-Levenshtein, max distance = 1

All four operations (insert, delete, substitute, transpose) each cost exactly 1.0. Only words within 1 edit are considered.

```text
cost(insert)     = 1.0
cost(delete)     = 1.0
cost(substitute) = 1.0
cost(transpose)  = 1.0

candidates : words where dist(typed, word) ≤ 1
edit_pen   = dist / 1
```

Good for catching single-character slips — a missing letter, one wrong key, or two swapped letters. Won't recover from larger mistakes.

---

### S2 — Standard Damerau-Levenshtein, max distance = 2

Same uniform costs as S1 but the threshold is relaxed to 2 edits. This catches the majority of real-world typos.

```text
cost(insert)     = 1.0
cost(delete)     = 1.0
cost(substitute) = 1.0
cost(transpose)  = 1.0

candidates : words where dist(typed, word) ≤ 2
edit_pen   = dist / 2
```

The wider net means more candidates compete for the top spots, so the language model or frequency term becomes more important for ranking.

---

### S3 — Weighted Damerau-Levenshtein, max distance = 2

Same threshold as S2, but operations no longer cost the same. Transpositions are cheaper (0.5) because letter swaps are physically common. For substitutions, you can optionally factor in keyboard layout.

#### S3 + Operation weighting

```text
cost(insert)     = 1.0
cost(delete)     = 1.0
cost(substitute) = 1.0
cost(transpose)  = 0.5   ← swapping two adjacent letters is "half a mistake"

candidates : words where weighted_dist(typed, word) ≤ 2
edit_pen   = weighted_dist / 2
```

#### S3 + Keyboard weighting

```text
cost(insert)     = 1.0
cost(delete)     = 1.0
cost(transpose)  = 0.5
cost(substitute) = 0.5   if the two keys are adjacent on QWERTY (Euclidean distance ≤ 1.5)
                 = 1.0   otherwise

candidates : words where weighted_dist(typed, word) ≤ 2
edit_pen   = weighted_dist / 2
```

The keyboard distances are computed from the physical QWERTY layout. For example, `a` → `s` costs 0.5 (neighbours), but `a` → `p` costs 1.0 (far apart). This reflects the reality that adjacent-key substitutions happen much more often by accident.

## Combining with a Language Model

By default, spell correction doesn't use any language model — it just looks at edit distance and word frequency. But you can pair it with one of the trained language models to get context-aware suggestions.

**No model** — purely edit-distance + frequency. Suggestions are the same regardless of what came before. Simple and fast.

**N-gram** — uses the n-gram language model to score candidates based on the words that came before. For example, after typing `"the quick brown"`, it knows `"fox"` is more likely than `"fax"` even if both are equally close to what you typed.

**Transformer** — uses the trained transformer model for the same purpose, but with a much richer understanding of context. It considers the whole sentence, not just the last few words. Generally gives the best suggestions, but is slower.

When you pick N-gram or Transformer as the language model for spell correction, it takes over the model selection for that session — the prediction panel and the spell correction panel both use the same underlying model.

## Quick Summary

| Setting | What it means |
| --- | --- |
| S1 | Only fix single-character mistakes |
| S2 | Fix up to 2 mistakes (recommended) |
| S3 | Fix up to 2 mistakes, with smarter cost weighting |
| Operation weighting | Swaps cost half as much as inserts/deletes |
| Keyboard weighting | Nearby keys cost less than distant ones |
| No model | Score by edit distance + frequency only |
| N-gram | Add sentence context via n-gram model |
| Transformer | Add sentence context via transformer model |
