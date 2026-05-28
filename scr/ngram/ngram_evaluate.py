import argparse
import json
import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ngram_model import NGramModel


def load_preprocessed_sentences(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    tokenized_sentences = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            tokens = line.split()

            if tokens:
                tokenized_sentences.append(tokens)

    return tokenized_sentences


def generate_lambda_grid(max_n_gram=4, step=0.1):
    units = int(round(1.0 / step))

    def generate_compositions(total, parts):
        if parts == 1:
            yield [total]
        else:
            for i in range(total + 1):
                for rest in generate_compositions(total - i, parts - 1):
                    yield [i] + rest

    for composition in generate_compositions(units, max_n_gram):
        lambdas = {}

        for n, value in enumerate(composition, start=1):
            lambdas[n] = value / units

        yield lambdas


def target_in_top_k_exact(model, candidates, scores, target_word, top_k):
    scored_candidates = zip(candidates, scores)

    top_candidates = heapq.nsmallest(
        top_k,
        scored_candidates,
        key=lambda item: (
            -item[1],
            -model.word_counts.get(item[0], 0),
            len(item[0]),
            item[0],
        ),
    )

    return any(word == target_word for word, score in top_candidates)


def tune_lambdas_one_validation_pass(
    model,
    val_sentences,
    top_k=5,
    grid_step=0.1,
    max_eval_sentences=1000,
    include_empty_prefix=False,
):
    if max_eval_sentences is not None:
        val_sentences = val_sentences[:max_eval_sentences]

    lambda_grid = list(generate_lambda_grid(model.max_n_gram, step=grid_step))
    n_values = list(range(1, model.max_n_gram + 1))

    lambda_matrix = np.array(
        [[lambdas[n] for n in n_values] for lambdas in lambda_grid],
        dtype=np.float64,
    )

    num_lambdas = len(lambda_grid)

    print()
    print("Number of lambda combinations:", num_lambdas)
    print("Number of validation sentences used:", len(val_sentences))
    print()

    total_words = 0
    total_characters = 0

    saved_keystrokes = np.zeros(num_lambdas, dtype=np.float64)
    successful_words = np.zeros(num_lambdas, dtype=np.float64)
    appeared_words = np.zeros(num_lambdas, dtype=np.float64)
    total_chars_typed_until_suggestion = np.zeros(num_lambdas, dtype=np.float64)

    for sentence in tqdm(val_sentences, desc="One-pass lambda tuning"):
        sentence = [word.lower().strip() for word in sentence if word.strip()]

        for i, target_word in enumerate(sentence):
            if not target_word:
                continue

            context = sentence[:i]

            total_words += 1
            total_characters += len(target_word)

            found = np.zeros(num_lambdas, dtype=bool)

            chars_typed_when_found = np.full(
                num_lambdas,
                len(target_word),
                dtype=np.int32,
            )

            if target_word not in model.vocab:
                total_chars_typed_until_suggestion += len(target_word)
                continue

            if include_empty_prefix:
                prefix_lengths = range(0, len(target_word) + 1)
            else:
                prefix_lengths = range(1, len(target_word) + 1)

            for chars_typed in prefix_lengths:
                if found.all():
                    break

                prefix = target_word[:chars_typed]
                candidates = model.get_candidates(prefix)

                if not candidates:
                    continue

                base_probabilities = np.array(
                    [
                        [
                            model.probability(context, candidate_word, n=n)
                            for n in n_values
                        ]
                        for candidate_word in candidates
                    ],
                    dtype=np.float64,
                )

                not_found_indices = np.where(~found)[0]

                for lambda_index in not_found_indices:
                    lambdas_vector = lambda_matrix[lambda_index]
                    scores = base_probabilities @ lambdas_vector

                    if target_in_top_k_exact(
                        model=model,
                        candidates=candidates,
                        scores=scores,
                        target_word=target_word,
                        top_k=top_k,
                    ):
                        found[lambda_index] = True
                        chars_typed_when_found[lambda_index] = chars_typed

            word_saved = len(target_word) - chars_typed_when_found
            word_saved = np.maximum(word_saved, 0)

            saved_keystrokes += word_saved
            appeared_words += found.astype(np.float64)
            successful_words += (word_saved > 0).astype(np.float64)
            total_chars_typed_until_suggestion += chars_typed_when_found

    all_results = []

    for lambda_index, lambdas in enumerate(lambda_grid):
        saved_keystroke_ratio = (
            saved_keystrokes[lambda_index] / total_characters
            if total_characters > 0
            else 0.0
        )

        success_rate = (
            successful_words[lambda_index] / total_words
            if total_words > 0
            else 0.0
        )

        appeared_rate = (
            appeared_words[lambda_index] / total_words
            if total_words > 0
            else 0.0
        )

        mean_chars_typed = (
            total_chars_typed_until_suggestion[lambda_index] / total_words
            if total_words > 0
            else 0.0
        )

        result = {
            "lambda_index": lambda_index,
            "lambdas": lambdas,
            "saved_keystrokes": int(saved_keystrokes[lambda_index]),
            "saved_keystroke_ratio": float(saved_keystroke_ratio),
            "success_rate": float(success_rate),
            "appeared_rate": float(appeared_rate),
            "mean_chars_typed_until_suggestion": float(mean_chars_typed),
            "top_k": top_k,
            "include_empty_prefix": include_empty_prefix,
        }

        all_results.append(result)

    all_results = sorted(
        all_results,
        key=lambda item: item["saved_keystroke_ratio"],
        reverse=True,
    )

    best_result = all_results[0]
    best_lambdas = best_result["lambdas"]
    best_lambda_index = best_result["lambda_index"]

    best_metrics = {
        "total_words": total_words,
        "total_characters": total_characters,
        "saved_keystrokes": int(saved_keystrokes[best_lambda_index]),
        "saved_keystroke_ratio": best_result["saved_keystroke_ratio"],
        "success_rate": best_result["success_rate"],
        "appeared_rate": best_result["appeared_rate"],
        "mean_chars_typed_until_suggestion": best_result["mean_chars_typed_until_suggestion"],
        "top_k": top_k,
        "lambdas": best_lambdas,
        "include_empty_prefix": include_empty_prefix,
    }

    return best_lambdas, best_metrics, all_results


def evaluate_saved_keystrokes(
    model,
    tokenized_sentences,
    lambdas,
    top_k=5,
    max_eval_sentences=None,
    include_empty_prefix=False,
):
    if max_eval_sentences is not None:
        tokenized_sentences = tokenized_sentences[:max_eval_sentences]

    total_words = 0
    total_characters = 0
    saved_keystrokes = 0

    successful_words = 0
    appeared_words = 0
    total_chars_typed_until_suggestion = 0

    top_k_correct_words = 0

    for sentence in tqdm(tokenized_sentences, desc="Evaluating validation"):
        sentence = [word.lower().strip() for word in sentence if word.strip()]

        for i, target_word in enumerate(sentence):
            if not target_word:
                continue

            context = sentence[:i]

            total_words += 1
            total_characters += len(target_word)

            target_in_vocab = target_word in model.vocab
            empty_prefix_suggestions = []

            if target_in_vocab:
                empty_prefix_suggestions = model.predict_interpolated(
                    context=context,
                    prefix="",
                    top_k=top_k,
                    lambdas=lambdas,
                    include_scores=False,
                )

                if target_word in empty_prefix_suggestions:
                    top_k_correct_words += 1

            found = False
            chars_typed_when_found = len(target_word)

            if not target_in_vocab:
                total_chars_typed_until_suggestion += chars_typed_when_found
                continue

            if include_empty_prefix:
                prefix_lengths = range(0, len(target_word) + 1)
            else:
                prefix_lengths = range(1, len(target_word) + 1)

            for chars_typed in prefix_lengths:
                if chars_typed == 0:
                    suggestions = empty_prefix_suggestions
                else:
                    prefix = target_word[:chars_typed]

                    suggestions = model.predict_interpolated(
                        context=context,
                        prefix=prefix,
                        top_k=top_k,
                        lambdas=lambdas,
                        include_scores=False,
                    )

                if target_word in suggestions:
                    found = True
                    chars_typed_when_found = chars_typed
                    break

            if found:
                appeared_words += 1

            word_saved_keystrokes = max(
                0,
                len(target_word) - chars_typed_when_found,
            )

            saved_keystrokes += word_saved_keystrokes
            total_chars_typed_until_suggestion += chars_typed_when_found

            if word_saved_keystrokes > 0:
                successful_words += 1

    saved_keystroke_ratio = (
        saved_keystrokes / total_characters
        if total_characters > 0
        else 0.0
    )

    success_rate = (
        successful_words / total_words
        if total_words > 0
        else 0.0
    )

    appeared_rate = (
        appeared_words / total_words
        if total_words > 0
        else 0.0
    )

    top_k_accuracy = (
        top_k_correct_words / total_words
        if total_words > 0
        else 0.0
    )

    mean_chars_typed = (
        total_chars_typed_until_suggestion / total_words
        if total_words > 0
        else 0.0
    )

    metrics = {
        "total_words": total_words,
        "total_characters": total_characters,
        "saved_keystrokes": saved_keystrokes,
        "saved_keystroke_ratio": saved_keystroke_ratio,
        "top_k_accuracy": top_k_accuracy,
        "top_k_correct_words": top_k_correct_words,
        "success_rate": success_rate,
        "appeared_rate": appeared_rate,
        "mean_chars_typed_until_suggestion": mean_chars_typed,
        "top_k": top_k,
        "lambdas": lambdas,
        "include_empty_prefix": include_empty_prefix,
    }

    return metrics


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def plot_best_lambdas(best_lambdas, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_values = sorted(best_lambdas.keys())
    weights = [best_lambdas[n] for n in n_values]

    plt.figure(figsize=(7, 5))
    plt.bar([str(n) + "-gram" for n in n_values], weights)
    plt.xlabel("Model")
    plt.ylabel("Interpolation weight")
    plt.title("Best interpolation weights found on validation set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_top_lambda_results(all_results, save_path, top_n=10):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    top_results = all_results[:top_n]

    labels = []
    scores = []

    for result in top_results:
        lambdas = result["lambdas"]

        label = ", ".join(
            f"{lambdas[n]:.2f}" for n in sorted(lambdas.keys())
        )

        labels.append(label)
        scores.append(result["saved_keystroke_ratio"])

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.xlabel("Lambdas: unigram, bigram, trigram, 4-gram")
    plt.ylabel("Saved keystroke ratio")
    plt.title(f"Top {top_n} lambda settings on validation set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_metrics(title, metrics):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print("Total words:", metrics["total_words"])
    print("Total characters:", metrics["total_characters"])
    print("Saved keystrokes:", metrics["saved_keystrokes"])
    print("Saved keystroke ratio:", round(metrics["saved_keystroke_ratio"], 4))
    print("Top-k accuracy:", round(metrics["top_k_accuracy"], 4))
    print("Top-k correct words:", metrics["top_k_correct_words"])
    print("Success rate:", round(metrics["success_rate"], 4))
    print("Prefix appeared rate:", round(metrics["appeared_rate"], 4))
    print("Mean chars typed until suggestion:", round(metrics["mean_chars_typed_until_suggestion"], 4))
    print("Top-k:", metrics["top_k"])
    print("Lambdas:", metrics["lambdas"])
    print("Include empty prefix:", metrics["include_empty_prefix"])


def parse_args():
    default_project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Tune n-gram interpolation lambdas on validation data."
    )

    parser.add_argument(
        "--project_root",
        type=Path,
        default=default_project_root,
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--val_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--best_lambdas_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--validation_results_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--best_lambdas_plot_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--top_lambdas_plot_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--grid_step",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max_val_sentences",
        type=int,
        help="Use 0 or a negative value to evaluate all validation sentences.",
    )

    parser.add_argument(
        "--include_empty_prefix",
        action="store_true",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root

    model_path = args.model_path or project_root / "models/ngram/Wikitext2_ngram_model.pkl"
    val_path = args.val_path or project_root / "scr/data/wikitext_2/wikitext2_val.txt"

    best_lambdas_path = (
        args.best_lambdas_path
        or project_root / "results/metrics/best_ngram_lambdasWikitext2.json"
    )

    validation_results_path = (
        args.validation_results_path
        or project_root / "results/metrics/ngram_validation_resultsWikitext2.json"
    )

    best_lambdas_plot_path = (
        args.best_lambdas_plot_path
        or project_root / "results/plots/best_ngram_lambdasWikitext2.png"
    )

    top_lambdas_plot_path = (
        args.top_lambdas_plot_path
        or project_root / "results/plots/top_ngram_lambdasWikitext2.png"
    )

    top_k = args.top_k
    grid_step = args.grid_step

    max_val_sentences = args.max_val_sentences
    if max_val_sentences is not None and max_val_sentences <= 0:
        max_val_sentences = None

    include_empty_prefix = args.include_empty_prefix
    make_plots = not args.no_plots

    print("Project root:", project_root)
    print("Model path:", model_path)
    print("Validation path:", val_path)
    print("Best lambdas path:", best_lambdas_path)
    print("Validation results path:", validation_results_path)
    print("Top-k:", top_k)
    print("Grid step:", grid_step)
    print("Max validation sentences:", max_val_sentences)
    print("Include empty prefix:", include_empty_prefix)

    print()
    print("Loading model...")
    model = NGramModel.load(model_path)
    print(model)

    print()
    print("Loading validation data...")
    val_sentences = load_preprocessed_sentences(val_path)
    print("Number of validation sentences:", len(val_sentences))

    best_lambdas, tuning_metrics, all_results = tune_lambdas_one_validation_pass(
        model=model,
        val_sentences=val_sentences,
        top_k=top_k,
        grid_step=grid_step,
        max_eval_sentences=max_val_sentences,
        include_empty_prefix=include_empty_prefix,
    )

    validation_metrics = evaluate_saved_keystrokes(
        model=model,
        tokenized_sentences=val_sentences,
        lambdas=best_lambdas,
        top_k=top_k,
        max_eval_sentences=max_val_sentences,
        include_empty_prefix=include_empty_prefix,
    )

    validation_result = {
        "best_lambdas": best_lambdas,
        "tuning_best_metrics": tuning_metrics,
        "best_validation_metrics": validation_metrics,
        "all_validation_results": all_results,
    }

    save_json(validation_result, best_lambdas_path)
    save_json(validation_result, validation_results_path)

    print_metrics("Best validation result", validation_metrics)

    print()
    print("Saved best lambdas to:")
    print(best_lambdas_path)

    print()
    print("Saved validation results to:")
    print(validation_results_path)

    if make_plots:
        plot_best_lambdas(best_lambdas, best_lambdas_plot_path)
        plot_top_lambda_results(all_results, top_lambdas_plot_path, top_n=10)

        print()
        print("Saved plots:")
        print(best_lambdas_plot_path)
        print(top_lambdas_plot_path)


if __name__ == "__main__":
    main()
