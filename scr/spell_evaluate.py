"""
Evaluate the SpellCorrector on noisy input.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "transformer"))
sys.path.insert(0, str(CURRENT_DIR / "ngram"))

from spell_corrector import SpellCorrector, introduce_typo

TOP_K_VALUES = [1, 2, 3, 4]
MAX_TOP_K = max(TOP_K_VALUES)
_STRIP_CHARS = ".,!?;:\"'()[]{}@-"


def load_sentences(path, max_sentences=None):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.lower().strip().split()
            if tokens:
                sentences.append(tokens)
                if max_sentences and len(sentences) >= max_sentences:
                    break
    return sentences


def load_github_typos(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0].lower(), parts[1].lower()))
    return pairs




def evaluate_synthetic(corrector, sentences, max_edit_dist=1):
    total_words = 0
    total_chars = 0
    saved_by_k = {k: 0 for k in TOP_K_VALUES}
    correct_by_k = {k: 0 for k in TOP_K_VALUES}

    for sentence in tqdm(sentences, desc="Evaluating (synthetic noise)"):
        sentence = [w for w in sentence if w]

        for i, raw_word in enumerate(sentence):
            target_word = raw_word.strip(_STRIP_CHARS)
            if not target_word or target_word not in corrector.word_vocab_set:
                continue

            corrupted = introduce_typo(target_word, max_edit_dist=max_edit_dist)
            if corrupted == target_word:
                continue

            context = sentence[:i]
            total_words += 1
            total_chars += len(target_word)

            found = {k: False for k in TOP_K_VALUES}
            chars_found = {k: len(target_word) for k in TOP_K_VALUES}

            for chars_typed in range(0, len(corrupted) + 1):
                if all(found.values()):
                    break
                prefix = corrupted[:chars_typed]
                suggestions = corrector.predict(
                    context, prefix=prefix, top_k=MAX_TOP_K, include_scores=False
                )
                if isinstance(suggestions[0], tuple) if suggestions and not isinstance(suggestions[0], str) else False:
                    suggestions = [w for w, _ in suggestions]

                for j, word in enumerate(suggestions, start=1):
                    if word == target_word:
                        for k in TOP_K_VALUES:
                            if not found[k] and j <= k:
                                found[k] = True
                                chars_found[k] = chars_typed
                        break

            for k in TOP_K_VALUES:
                if found[k]:
                    correct_by_k[k] += 1
                saved_by_k[k] += max(0, len(target_word) - chars_found[k])

    return _build_results(total_words, total_chars, correct_by_k, saved_by_k)


def evaluate_github(corrector, sentences, typo_pairs):
    typo_map = {typo: correction for typo, correction in typo_pairs}

    total_words = 0
    total_chars = 0
    saved_by_k = {k: 0 for k in TOP_K_VALUES}
    correct_by_k = {k: 0 for k in TOP_K_VALUES}

    for sentence in tqdm(sentences, desc="Evaluating (GitHub typos)"):
        sentence = [w for w in sentence if w]

        for i, raw_word in enumerate(sentence):
            target_word = raw_word.strip(_STRIP_CHARS)
            if not target_word:
                continue

            if target_word not in typo_map:
                continue
            corrupted = typo_map[target_word]
            if not corrupted or corrupted == target_word:
                continue
            if target_word not in corrector.word_vocab_set:
                continue

            context = sentence[:i]
            total_words += 1
            total_chars += len(target_word)

            found = {k: False for k in TOP_K_VALUES}
            chars_found = {k: len(target_word) for k in TOP_K_VALUES}

            for chars_typed in range(0, len(corrupted) + 1):
                if all(found.values()):
                    break
                prefix = corrupted[:chars_typed]
                suggestions = corrector.predict(
                    context, prefix=prefix, top_k=MAX_TOP_K, include_scores=False
                )
                if suggestions and not isinstance(suggestions[0], str):
                    suggestions = [w for w, _ in suggestions]

                for j, word in enumerate(suggestions, start=1):
                    if word == target_word:
                        for k in TOP_K_VALUES:
                            if not found[k] and j <= k:
                                found[k] = True
                                chars_found[k] = chars_typed
                        break

            for k in TOP_K_VALUES:
                if found[k]:
                    correct_by_k[k] += 1
                saved_by_k[k] += max(0, len(target_word) - chars_found[k])

    return _build_results(total_words, total_chars, correct_by_k, saved_by_k)


def _build_results(total_words, total_chars, correct_by_k, saved_by_k):
    results = {}
    for k in TOP_K_VALUES:
        results[k] = {
            "top_k": k,
            "total_words": total_words,
            "total_characters": total_chars,
            "top_k_correct_words": correct_by_k[k],
            "top_k_accuracy": correct_by_k[k] / total_words if total_words > 0 else 0.0,
            "saved_keystrokes": saved_by_k[k],
            "saved_keystroke_ratio": saved_by_k[k] / total_chars if total_chars > 0 else 0.0,
        }
    return results


def print_results(results, label):
    for k in TOP_K_VALUES:
        m = results[k]
        print()
        print("=" * 60)
        print(f"SpellCorrector — {label} — top-{k}")
        print("=" * 60)
        print(f"Total words evaluated : {m['total_words']:,}")
        print(f"Total characters      : {m['total_characters']:,}")
        print(f"Top-{k} accuracy      : {m['top_k_accuracy']:.4f}")
        print(f"Saved keystrokes      : {m['saved_keystrokes']:,}")
        print(f"Saved-keystroke ratio : {m['saved_keystroke_ratio']:.4f}")



def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["transformer", "ngram"], default="transformer")
    p.add_argument("--dataset",     choices=["tinystories", "wikitext2"], default="wikitext2")
    p.add_argument("--noise_mode",  choices=["synthetic", "github"], default="synthetic")
    p.add_argument("--max_edit_dist", type=int, default=2)
    p.add_argument("--lambda_edit", type=float, default=0.5)
    p.add_argument("--mu_freq",     type=float, default=0.1)
    p.add_argument("--weight_mode", choices=["uniform", "operation", "keyboard"],
                   default="uniform")
    p.add_argument("--max_test_sentences", type=int, default=500)
    p.add_argument("--github_typo_path", default=None)
    p.add_argument("--checkpoint_path",  default=None)
    p.add_argument("--tokenizer_path",   default=None)
    p.add_argument("--ngram_model_path", default=None)
    p.add_argument("--results_path",     default=None)
    p.add_argument("--device",           default="cpu")
    p.add_argument("--project_root",     default=str(project_root))
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    if args.dataset == "wikitext2":
        data_dir   = project_root / "scr/data/wikitext_2_transformer"
        model_dir  = project_root / "models/transformer/wikitext2"
        ngram_pkl  = project_root / "models/ngram/wikitext2_ngram_model.pkl"
        prefix     = "wikitext2"
    else:
        data_dir   = project_root / "scr/data/tiny_stories_transformer"
        model_dir  = project_root / "models/transformer/tinystories"
        ngram_pkl  = project_root / "models/ngram/Tiny_stories_ngram_model.pkl"
        prefix     = "tinystories"

    test_path = data_dir / f"{prefix}_transformer_test.txt"


    if args.model == "transformer":
        from transformer_model import TransformerPredictor
        checkpoint = args.checkpoint_path or str(model_dir / "checkpoint_final.pt")
        tokenizer  = args.tokenizer_path  or str(model_dir / "tokenizer.json")
        predictor  = TransformerPredictor(checkpoint, tokenizer, device=args.device)
        vocab_source = str(ngram_pkl) if ngram_pkl.exists() else str(data_dir / f"{prefix}_transformer_train.txt")
        if ngram_pkl.exists():
            predictor.load_word_vocab_from_ngram(str(ngram_pkl))
        else:
            predictor.build_word_vocab_from_text(str(data_dir / f"{prefix}_transformer_train.txt"))
    else:
        from ngram_model import NGramModel
        ngram_path = args.ngram_model_path or str(ngram_pkl)
        predictor  = NGramModel.load(ngram_path)


    corrector = SpellCorrector(
        predictor,
        max_edit_dist=args.max_edit_dist,
        lambda_edit=args.lambda_edit,
        mu_freq=args.mu_freq,
        weight_mode=args.weight_mode,
    )
    print(f"SpellCorrector ready. Vocab: {len(corrector.word_vocab):,} words. "
          f"max_edit_dist={args.max_edit_dist}, λ={args.lambda_edit}, μ={args.mu_freq}, "
          f"weight_mode={args.weight_mode}")


    sentences = load_sentences(
        str(test_path),
        max_sentences=args.max_test_sentences if args.max_test_sentences > 0 else None,
    )
    print(f"Test sentences: {len(sentences):,}")


    if args.noise_mode == "github":
        if not args.github_typo_path:
            raise ValueError("--github_typo_path required for github noise mode")
        typo_pairs = load_github_typos(args.github_typo_path)
        print(f"GitHub typo pairs: {len(typo_pairs):,}")
        results = evaluate_github(corrector, sentences, typo_pairs)
        label = f"{args.dataset} | github-typos | edit≤{args.max_edit_dist} | {args.weight_mode}"
    else:
        results = evaluate_synthetic(corrector, sentences, max_edit_dist=args.max_edit_dist)
        label = f"{args.dataset} | synthetic-noise | edit≤{args.max_edit_dist} | {args.weight_mode}"

    print_results(results, label)


    results_path = args.results_path or str(
        project_root / f"results/metrics/{prefix}_spell_{args.model}_{args.noise_mode}_{args.weight_mode}_results.json"
    )
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "noise_mode": args.noise_mode,
            "max_edit_dist": args.max_edit_dist,
            "lambda_edit": args.lambda_edit,
            "mu_freq": args.mu_freq,
            "weight_mode": args.weight_mode,
            "results_by_top_k": results,
        }, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
