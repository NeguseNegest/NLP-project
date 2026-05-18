import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from ngram_model import NGramModel


TOP_K_VALUES = [1, 2, 3, 4]
MAX_TOP_K = max(TOP_K_VALUES)


def load_preprocessed_sentences(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    sentences = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()

            if tokens:
                sentences.append(tokens)

    return sentences


def parse_lambdas(lambda_string, max_n_gram):
    values = [float(value.strip()) for value in lambda_string.split(",")]

    if len(values) != max_n_gram:
        raise ValueError(f"Expected {max_n_gram} lambda values, got {len(values)}.")

    if any(value < 0 for value in values):
        raise ValueError("Lambda values must be non-negative.")

    total = sum(values)

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Lambda values must sum to 1. Got {total}.")

    return {n: value for n, value in enumerate(values, start=1)}


def get_rank(suggestions, target_word):
    for index, word in enumerate(suggestions, start=1):
        if word == target_word:
            return index

    return None


def evaluate_test(
    model,
    tokenized_sentences,
    lambdas,
    max_eval_sentences=None,
    include_empty_prefix=False,
):
    if max_eval_sentences is not None:
        tokenized_sentences = tokenized_sentences[:max_eval_sentences]

    total_words = 0
    total_characters = 0
    total_log_probability = 0.0

    saved_keystrokes_by_k = {k: 0 for k in TOP_K_VALUES}
    top_k_correct_by_k = {k: 0 for k in TOP_K_VALUES}

    for sentence in tqdm(tokenized_sentences, desc="Evaluating test"):
        sentence = [word.lower().strip() for word in sentence if word.strip()]

        for i, target_word in enumerate(sentence):
            context = sentence[:i]

            total_words += 1
            total_characters += len(target_word)

            probability = model.interpolated_probability(
                context=context,
                word=target_word,
                lambdas=lambdas,
            )

            total_log_probability += math.log(max(probability, 1e-300))

            if target_word not in model.vocab:
                continue

            empty_suggestions = model.predict_interpolated(
                context=context,
                prefix="",
                top_k=MAX_TOP_K,
                lambdas=lambdas,
                include_scores=False,
            )

            empty_rank = get_rank(empty_suggestions, target_word)

            if empty_rank is not None:
                for k in TOP_K_VALUES:
                    if empty_rank <= k:
                        top_k_correct_by_k[k] += 1

            found = {k: False for k in TOP_K_VALUES}
            chars_typed_when_found = {k: len(target_word) for k in TOP_K_VALUES}

            if include_empty_prefix:
                prefix_lengths = range(0, len(target_word) + 1)
            else:
                prefix_lengths = range(1, len(target_word) + 1)

            for chars_typed in prefix_lengths:
                if all(found.values()):
                    break

                if chars_typed == 0:
                    suggestions = empty_suggestions
                else:
                    prefix = target_word[:chars_typed]

                    suggestions = model.predict_interpolated(
                        context=context,
                        prefix=prefix,
                        top_k=MAX_TOP_K,
                        lambdas=lambdas,
                        include_scores=False,
                    )

                rank = get_rank(suggestions, target_word)

                if rank is None:
                    continue

                for k in TOP_K_VALUES:
                    if not found[k] and rank <= k:
                        found[k] = True
                        chars_typed_when_found[k] = chars_typed

            for k in TOP_K_VALUES:
                saved = len(target_word) - chars_typed_when_found[k]
                saved_keystrokes_by_k[k] += max(0, saved)

    cross_entropy = -total_log_probability / total_words if total_words > 0 else 0.0
    perplexity = math.exp(cross_entropy) if total_words > 0 else 0.0

    results = {}

    for k in TOP_K_VALUES:
        saved_keystrokes = saved_keystrokes_by_k[k]
        top_k_correct_words = top_k_correct_by_k[k]

        saved_keystroke_ratio = (
            saved_keystrokes / total_characters
            if total_characters > 0
            else 0.0
        )

        top_k_accuracy = (
            top_k_correct_words / total_words
            if total_words > 0
            else 0.0
        )

        results[k] = {
            "top_k": k,
            "total_words": total_words,
            "total_characters": total_characters,
            "saved_keystrokes": saved_keystrokes,
            "saved_keystroke_ratio": saved_keystroke_ratio,
            "top_k_correct_words": top_k_correct_words,
            "top_k_accuracy": top_k_accuracy,
            "cross_entropy": cross_entropy,
            "perplexity": perplexity,
            "lambdas": lambdas,
            "include_empty_prefix": include_empty_prefix,
        }

    return results


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def print_results(results):
    for k in TOP_K_VALUES:
        metrics = results[k]

        print()
        print("=" * 60)
        print(f"Final test result, top-{k}")
        print("=" * 60)
        print("Total words:", metrics["total_words"])
        print("Total characters:", metrics["total_characters"])
        print("Saved keystrokes:", metrics["saved_keystrokes"])
        print("Saved-keystroke ratio:", round(metrics["saved_keystroke_ratio"], 4))
        print("Top-k correct words:", metrics["top_k_correct_words"])
        print("Top-k accuracy:", round(metrics["top_k_accuracy"], 4))
        print("Cross-entropy:", round(metrics["cross_entropy"], 4))
        print("Perplexity:", round(metrics["perplexity"], 4))
        print("Lambdas:", metrics["lambdas"])
        print("Include empty prefix:", metrics["include_empty_prefix"])


def parse_args():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=project_root)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--test_path", type=Path, default=None)
    parser.add_argument("--test_results_path", type=Path, default=None)
    parser.add_argument("--lambdas", type=str, required=True)
    parser.add_argument("--max_test_sentences", type=int, default=0)
    parser.add_argument("--include_empty_prefix", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root

    model_path = (
        args.model_path
        or project_root / "models/ngram/Wikitext2_ngram_model.pkl"
    )

    test_path = (
        args.test_path
        or project_root / "scr/data/wikitext_2/wikitext2_test.txt"
    )

    test_results_path = (
        args.test_results_path
        or project_root / "results/metrics/ngram_test_results_top_1_to_4_Wikitext2.json"
    )

    max_test_sentences = args.max_test_sentences

    if max_test_sentences <= 0:
        max_test_sentences = None

    print("Model path:", model_path)
    print("Test path:", test_path)
    print("Results path:", test_results_path)
    print("Top-k values:", TOP_K_VALUES)
    print("Max test sentences:", max_test_sentences)
    print("Include empty prefix:", args.include_empty_prefix)

    print()
    print("Loading model...")
    model = NGramModel.load(model_path)
    print(model)

    lambdas = parse_lambdas(args.lambdas, model.max_n_gram)

    print()
    print("Using lambdas:", lambdas)

    print()
    print("Loading test data...")
    test_sentences = load_preprocessed_sentences(test_path)
    print("Number of test sentences:", len(test_sentences))

    results = evaluate_test(
        model=model,
        tokenized_sentences=test_sentences,
        lambdas=lambdas,
        max_eval_sentences=max_test_sentences,
        include_empty_prefix=args.include_empty_prefix,
    )

    output = {
        "top_k_values": TOP_K_VALUES,
        "lambdas": lambdas,
        "include_empty_prefix": args.include_empty_prefix,
        "max_test_sentences": max_test_sentences,
        "model_path": str(model_path),
        "test_path": str(test_path),
        "results_by_top_k": results,
    }

    save_json(output, test_results_path)
    print_results(results)

    print()
    print("Saved test results to:")
    print(test_results_path)


if __name__ == "__main__":
    main()