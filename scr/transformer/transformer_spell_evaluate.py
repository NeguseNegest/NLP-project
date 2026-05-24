"""
Evaluate Transformer spell correction on fixed corrupted-word datasets.

This file is intentionally separate from transformer_evaluate.py because it
evaluates misspelled final-word correction rather than clean next-word
prediction.

Examples:
    # Run all datasets and all spell strategies
    python scr/transformer/transformer_spell_evaluate.py

    # Quick smoke test
    python scr/transformer/transformer_spell_evaluate.py --dataset tinystories --strategy s1 --max_examples 20

Outputs:
    results/metrics/transformer_spell/
    results/plots/transformer_spell/
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "nlp_project_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
NGRAM_DIR = SRC_DIR / "ngram"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(NGRAM_DIR))

from spell_corrector import SpellCorrector
from transformer_model import TransformerPredictor


TOP_K_VALUES = [1, 2, 3, 4]
MAX_TOP_K = max(TOP_K_VALUES)
MAIN_TOP_K = 3

DATASET_CONFIGS = {
    "tinystories": {
        "model_dir": "models/transformer/tinystories",
        "ngram_model_candidates": [
            "models/ngram/Tiny_stories_ngram_model.pkl",
            "models/ngram/tinystories_ngram_model.pkl",
        ],
        "train_text": "scr/data/tiny_stories_transformer/tinystories_transformer_train.txt",
        "spell_eval_dir": "scr/data/spell_eval/tinystories",
    },
    "wikitext2": {
        "model_dir": "models/transformer/wikitext2",
        "ngram_model_candidates": [
            "models/ngram/Wikitext2_ngram_model.pkl",
            "models/ngram/wikitext2_ngram_model.pkl",
        ],
        "train_text": "scr/data/wikitext_2_transformer/wikitext2_transformer_train.txt",
        "spell_eval_dir": "scr/data/spell_eval/wikitext2",
    },
}

STRATEGY_CONFIGS = {
    "s1": {
        "data_file": "test_edit1.jsonl",
        "max_edit_dist": 1,
        "weight_mode": "uniform",
        "description": "S1: exact 1-edit data, uniform Damerau-Levenshtein",
    },
    "s2": {
        "data_file": "test_edit2.jsonl",
        "max_edit_dist": 2,
        "weight_mode": "uniform",
        "description": "S2: exact 2-edit data, uniform Damerau-Levenshtein",
    },
    "s3": {
        "data_file": "test_edit2.jsonl",
        "max_edit_dist": 2,
        "weight_mode": "operation",
        "description": "S3: exact 2-edit data, operation-weighted Damerau-Levenshtein",
    },
}

CONTEXT_BINS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, None, "51+"),
]

EDIT_OPERATIONS = ["deletion", "insertion", "substitution", "transposition"]


def resolve_existing_path(project_root, candidates):
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    return project_root / candidates[0]


def load_jsonl(path, max_examples=0):
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def get_rank(suggestions, target_word):
    if suggestions and not isinstance(suggestions[0], str):
        suggestions = [word for word, _ in suggestions]

    for rank, word in enumerate(suggestions, start=1):
        if word == target_word:
            return rank
    return None


def context_bin(context_length):
    for low, high, label in CONTEXT_BINS:
        if high is None and context_length >= low:
            return label
        if high is not None and low <= context_length <= high:
            return label
    return "unknown"


def new_bucket():
    return {
        "total_examples": 0,
        "total_characters": 0,
        "correct_by_k": {k: 0 for k in TOP_K_VALUES},
        "saved_by_k": {k: 0 for k in TOP_K_VALUES},
    }


def update_bucket(bucket, target_len, found_by_k, saved_by_k):
    bucket["total_examples"] += 1
    bucket["total_characters"] += target_len

    for k in TOP_K_VALUES:
        if found_by_k[k]:
            bucket["correct_by_k"][k] += 1
        bucket["saved_by_k"][k] += saved_by_k[k]


def finalize_bucket(bucket):
    results = {}
    total_examples = bucket["total_examples"]
    total_characters = bucket["total_characters"]

    for k in TOP_K_VALUES:
        correct = bucket["correct_by_k"][k]
        saved = bucket["saved_by_k"][k]
        results[str(k)] = {
            "top_k": k,
            "total_examples": total_examples,
            "total_characters": total_characters,
            "top_k_correct_words": correct,
            "top_k_accuracy": correct / total_examples if total_examples else 0.0,
            "saved_keystrokes": saved,
            "saved_keystroke_ratio": saved / total_characters if total_characters else 0.0,
        }

    return results


def finalize_breakdowns(buckets):
    return {
        group: finalize_bucket(bucket)
        for group, bucket in sorted(buckets.items(), key=lambda item: item[0])
    }


def evaluate_example(corrector, example, start_chars_typed):
    context = [word.lower() for word in example["context"]]
    target = example["target"].lower()
    corrupted = example["corrupted"].lower()

    found_by_k = {k: False for k in TOP_K_VALUES}
    chars_found_by_k = {k: len(target) for k in TOP_K_VALUES}
    best_rank = None
    first_found_chars = None

    first_prefix_len = min(start_chars_typed, len(corrupted))

    for chars_typed in range(first_prefix_len, len(corrupted) + 1):
        if all(found_by_k.values()):
            break

        prefix = corrupted[:chars_typed]
        suggestions = corrector.predict(
            context,
            prefix=prefix,
            top_k=MAX_TOP_K,
            include_scores=False,
            use_lm=True,
        )
        rank = get_rank(suggestions, target)
        if rank is None:
            continue

        best_rank = rank if best_rank is None else min(best_rank, rank)
        if first_found_chars is None:
            first_found_chars = chars_typed

        for k in TOP_K_VALUES:
            if not found_by_k[k] and rank <= k:
                found_by_k[k] = True
                chars_found_by_k[k] = chars_typed

    saved_by_k = {
        k: max(0, len(target) - chars_found_by_k[k])
        for k in TOP_K_VALUES
    }

    return {
        "found_by_top_k": {str(k): found_by_k[k] for k in TOP_K_VALUES},
        "chars_typed_by_top_k": {str(k): chars_found_by_k[k] for k in TOP_K_VALUES},
        "saved_by_top_k": {str(k): saved_by_k[k] for k in TOP_K_VALUES},
        "best_observed_rank": best_rank,
        "first_found_chars": first_found_chars,
    }, found_by_k, saved_by_k


def evaluate_dataset_strategy(corrector, examples, dataset_name, strategy_name, start_chars_typed):
    aggregate = new_bucket()
    by_edit_type = defaultdict(new_bucket)
    by_edit_operation = defaultdict(new_bucket)
    by_context_bin = defaultdict(new_bucket)
    context_lengths = Counter()
    example_results = []
    skipped_oov = 0
    vocab_set = set(corrector.word_vocab)

    progress_label = f"Transformer spell {dataset_name} {strategy_name}"

    for example in tqdm(examples, desc=progress_label):
        target = example["target"].lower()
        if target not in vocab_set:
            skipped_oov += 1
            skipped_result = {
                **example,
                "skipped": True,
                "skip_reason": "target_not_in_word_vocab",
                "context_length": len(example["context"]),
                "target_length": len(target),
                "corrupted_length": len(example["corrupted"]),
            }
            example_results.append(skipped_result)
            continue

        eval_result, found_by_k, saved_by_k = evaluate_example(
            corrector,
            example,
            start_chars_typed=start_chars_typed,
        )
        target_len = len(target)
        context_length = len(example["context"])
        bin_label = context_bin(context_length)
        operations = example.get("edit_operations") or [example.get("edit_type", "unknown")]
        edit_type = example.get("edit_type", "+".join(operations))

        context_lengths[context_length] += 1
        update_bucket(aggregate, target_len, found_by_k, saved_by_k)
        update_bucket(by_edit_type[edit_type], target_len, found_by_k, saved_by_k)
        update_bucket(by_context_bin[bin_label], target_len, found_by_k, saved_by_k)

        for operation in sorted(set(operations)):
            update_bucket(by_edit_operation[operation], target_len, found_by_k, saved_by_k)

        example_results.append(
            {
                **example,
                "skipped": False,
                "context_length": context_length,
                "context_bin": bin_label,
                "target_length": target_len,
                "corrupted_length": len(example["corrupted"]),
                **eval_result,
            }
        )

    return {
        "dataset": dataset_name,
        "strategy": strategy_name,
        "loaded_examples": len(examples),
        "evaluated_examples": aggregate["total_examples"],
        "skipped_oov_examples": skipped_oov,
        "results_by_top_k": finalize_bucket(aggregate),
        "breakdowns": {
            "by_edit_type": finalize_breakdowns(by_edit_type),
            "by_edit_operation": finalize_breakdowns(by_edit_operation),
            "by_context_bin": finalize_breakdowns(by_context_bin),
        },
        "context_length_counts": dict(sorted(context_lengths.items())),
        "example_results": example_results,
    }


def print_run_summary(result):
    print()
    print("=" * 72)
    print(f"Transformer spell correction — {result['dataset']} — {result['strategy'].upper()}")
    print("=" * 72)
    print(f"Loaded examples       : {result['loaded_examples']:,}")
    print(f"Evaluated examples    : {result['evaluated_examples']:,}")
    print(f"Skipped OOV examples  : {result['skipped_oov_examples']:,}")

    for k in TOP_K_VALUES:
        metrics = result["results_by_top_k"][str(k)]
        print(
            f"Top-{k}: accuracy={metrics['top_k_accuracy']:.4f} | "
            f"saved ratio={metrics['saved_keystroke_ratio']:.4f} | "
            f"saved={metrics['saved_keystrokes']:,}"
        )


def metric_for_group(group_results, top_k, metric_name):
    top_k_results = group_results.get(str(top_k))
    if not top_k_results:
        return 0.0
    return top_k_results.get(metric_name, 0.0)


def make_bar_plot(labels, values, title, ylabel, output_path, rotation=0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, len(labels) * 0.9), 4.8))
    plt.bar(labels, values, color="#3b82f6")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1 if values and max(values) <= 1 else None)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_context_histogram(result, output_path):
    counts = {
        int(length): count
        for length, count in result["context_length_counts"].items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))

    if counts:
        lengths = sorted(counts)
        values = [counts[length] for length in lengths]
        plt.bar(lengths, values, color="#64748b")

    plt.title(
        f"{result['dataset']} {result['strategy'].upper()} - words before corrupted word"
    )
    plt.xlabel("Number of words before corrupted word")
    plt.ylabel("Example count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_run(result, plot_dir):
    dataset = result["dataset"]
    strategy = result["strategy"]
    prefix = f"transformer_spell_{dataset}_{strategy}"

    labels = [f"Top-{k}" for k in TOP_K_VALUES]
    accuracies = [
        result["results_by_top_k"][str(k)]["top_k_accuracy"]
        for k in TOP_K_VALUES
    ]
    saved_ratios = [
        result["results_by_top_k"][str(k)]["saved_keystroke_ratio"]
        for k in TOP_K_VALUES
    ]

    make_bar_plot(
        labels,
        accuracies,
        f"{dataset} {strategy.upper()} - top-k accuracy",
        "Accuracy",
        plot_dir / f"{prefix}_topk_accuracy.png",
    )
    make_bar_plot(
        labels,
        saved_ratios,
        f"{dataset} {strategy.upper()} - saved keystroke ratio",
        "Saved keystroke ratio",
        plot_dir / f"{prefix}_saved_keystroke_ratio.png",
    )
    plot_context_histogram(
        result,
        plot_dir / f"{prefix}_context_lengths.png",
    )

    operation_results = result["breakdowns"]["by_edit_operation"]
    operation_labels = [operation for operation in EDIT_OPERATIONS if operation in operation_results]
    if operation_labels:
        make_bar_plot(
            operation_labels,
            [
                metric_for_group(
                    operation_results[operation],
                    MAIN_TOP_K,
                    "saved_keystroke_ratio",
                )
                for operation in operation_labels
            ],
            f"{dataset} {strategy.upper()} - Top-{MAIN_TOP_K} saved ratio by edit operation",
            "Saved keystroke ratio",
            plot_dir / f"{prefix}_top{MAIN_TOP_K}_saved_ratio_by_operation.png",
            rotation=20,
        )
        make_bar_plot(
            operation_labels,
            [
                metric_for_group(
                    operation_results[operation],
                    MAIN_TOP_K,
                    "top_k_accuracy",
                )
                for operation in operation_labels
            ],
            f"{dataset} {strategy.upper()} - Top-{MAIN_TOP_K} accuracy by edit operation",
            "Accuracy",
            plot_dir / f"{prefix}_top{MAIN_TOP_K}_accuracy_by_operation.png",
            rotation=20,
        )

    context_results = result["breakdowns"]["by_context_bin"]
    context_labels = [label for _, _, label in CONTEXT_BINS if label in context_results]
    if context_labels:
        make_bar_plot(
            context_labels,
            [
                metric_for_group(
                    context_results[label],
                    MAIN_TOP_K,
                    "saved_keystroke_ratio",
                )
                for label in context_labels
            ],
            f"{dataset} {strategy.upper()} - Top-{MAIN_TOP_K} saved ratio by context length",
            "Saved keystroke ratio",
            plot_dir / f"{prefix}_top{MAIN_TOP_K}_saved_ratio_by_context_bin.png",
            rotation=20,
        )


def plot_dataset_comparison(dataset_name, results, plot_dir):
    if not results:
        return

    labels = [result["strategy"].upper() for result in results]
    accuracies = [
        result["results_by_top_k"][str(MAIN_TOP_K)]["top_k_accuracy"]
        for result in results
    ]
    saved_ratios = [
        result["results_by_top_k"][str(MAIN_TOP_K)]["saved_keystroke_ratio"]
        for result in results
    ]

    output_path = plot_dir / f"transformer_spell_{dataset_name}_top{MAIN_TOP_K}_strategy_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(labels)))
    width = 0.36

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(
        [x - width / 2 for x in x_positions],
        accuracies,
        width=width,
        label=f"Top-{MAIN_TOP_K} accuracy",
        color="#2563eb",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        saved_ratios,
        width=width,
        label=f"Top-{MAIN_TOP_K} saved ratio",
        color="#16a34a",
    )
    plt.xticks(x_positions, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{dataset_name} - Transformer spell strategy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_run_outputs(result, metrics_dir):
    dataset = result["dataset"]
    strategy = result["strategy"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    compact_result = {
        key: value
        for key, value in result.items()
        if key != "example_results"
    }

    result_path = metrics_dir / f"transformer_spell_{dataset}_{strategy}_results.json"
    examples_path = metrics_dir / f"transformer_spell_{dataset}_{strategy}_examples.jsonl"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(compact_result, f, indent=4)

    with examples_path.open("w", encoding="utf-8") as f:
        for example_result in result["example_results"]:
            json.dump(example_result, f, ensure_ascii=True)
            f.write("\n")

    return result_path, examples_path


def write_summary_outputs(all_results, metrics_dir):
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = metrics_dir / "transformer_spell_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {key: value for key, value in result.items() if key != "example_results"}
                for result in all_results
            ],
            f,
            indent=4,
        )

    summary_csv_path = metrics_dir / "transformer_spell_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "strategy",
            "top_k",
            "loaded_examples",
            "evaluated_examples",
            "skipped_oov_examples",
            "total_characters",
            "top_k_correct_words",
            "top_k_accuracy",
            "saved_keystrokes",
            "saved_keystroke_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            for k in TOP_K_VALUES:
                metrics = result["results_by_top_k"][str(k)]
                writer.writerow(
                    {
                        "dataset": result["dataset"],
                        "strategy": result["strategy"],
                        "top_k": k,
                        "loaded_examples": result["loaded_examples"],
                        "evaluated_examples": result["evaluated_examples"],
                        "skipped_oov_examples": result["skipped_oov_examples"],
                        "total_characters": metrics["total_characters"],
                        "top_k_correct_words": metrics["top_k_correct_words"],
                        "top_k_accuracy": metrics["top_k_accuracy"],
                        "saved_keystrokes": metrics["saved_keystrokes"],
                        "saved_keystroke_ratio": metrics["saved_keystroke_ratio"],
                    }
                )

    breakdown_csv_path = metrics_dir / "transformer_spell_breakdowns.csv"
    with breakdown_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "strategy",
            "breakdown",
            "group",
            "top_k",
            "total_examples",
            "total_characters",
            "top_k_correct_words",
            "top_k_accuracy",
            "saved_keystrokes",
            "saved_keystroke_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            for breakdown_name, groups in result["breakdowns"].items():
                for group, group_results in groups.items():
                    for k in TOP_K_VALUES:
                        metrics = group_results[str(k)]
                        writer.writerow(
                            {
                                "dataset": result["dataset"],
                                "strategy": result["strategy"],
                                "breakdown": breakdown_name,
                                "group": group,
                                "top_k": k,
                                "total_examples": metrics["total_examples"],
                                "total_characters": metrics["total_characters"],
                                "top_k_correct_words": metrics["top_k_correct_words"],
                                "top_k_accuracy": metrics["top_k_accuracy"],
                                "saved_keystrokes": metrics["saved_keystrokes"],
                                "saved_keystroke_ratio": metrics["saved_keystroke_ratio"],
                            }
                        )

    return summary_json_path, summary_csv_path, breakdown_csv_path


def parse_args():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Evaluate Transformer spell correction on fixed misspelling datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "tinystories", "wikitext2"],
        default="all",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "s1", "s2", "s3"],
        default="all",
    )
    parser.add_argument("--project_root", default=str(project_root))
    parser.add_argument("--spell_data_dir", default=None)
    parser.add_argument("--metrics_dir", default=None)
    parser.add_argument("--plot_dir", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--ngram_model_path", default=None)
    parser.add_argument("--vocab_train_text", default=None)
    parser.add_argument("--lambda_edit", type=float, default=0.5)
    parser.add_argument("--mu_freq", type=float, default=0.1)
    parser.add_argument(
        "--start_chars_typed",
        type=int,
        default=1,
        help="First corrupted-prefix length to evaluate. Use 1 for spell correction.",
    )
    parser.add_argument(
        "--s3_weight_mode",
        choices=["operation", "keyboard"],
        default="operation",
    )
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--device", default="cpu")

    return parser.parse_args()


def selected_items(selection, options):
    if selection == "all":
        return list(options)
    return [selection]


def build_predictor(args, dataset_name):
    project_root = Path(args.project_root)
    config = DATASET_CONFIGS[dataset_name]
    model_dir = project_root / config["model_dir"]

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else model_dir / "checkpoint_final.pt"
    )
    tokenizer_path = (
        Path(args.tokenizer_path)
        if args.tokenizer_path
        else model_dir / "tokenizer.json"
    )

    print()
    print(f"Loading Transformer for {dataset_name}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Tokenizer  : {tokenizer_path}")

    predictor = TransformerPredictor(
        str(checkpoint_path),
        str(tokenizer_path),
        device=args.device,
    )

    if args.ngram_model_path:
        vocab_path = Path(args.ngram_model_path)
        predictor.load_word_vocab_from_ngram(str(vocab_path))
    elif args.vocab_train_text:
        predictor.build_word_vocab_from_text(args.vocab_train_text)
    else:
        ngram_path = resolve_existing_path(
            project_root,
            config["ngram_model_candidates"],
        )
        if ngram_path.exists():
            predictor.load_word_vocab_from_ngram(str(ngram_path))
        else:
            predictor.build_word_vocab_from_text(str(project_root / config["train_text"]))

    print(f"Word vocabulary size: {len(predictor.word_vocab):,}")
    return predictor


def strategy_config(strategy_name, args):
    config = dict(STRATEGY_CONFIGS[strategy_name])
    if strategy_name == "s3":
        config["weight_mode"] = args.s3_weight_mode
        config["description"] = (
            f"S3: exact 2-edit data, {args.s3_weight_mode}-weighted Damerau-Levenshtein"
        )
    return config


def spell_eval_path(args, dataset_name, strategy_name):
    project_root = Path(args.project_root)
    strategy = strategy_config(strategy_name, args)

    if args.spell_data_dir:
        base_dir = Path(args.spell_data_dir) / dataset_name
    else:
        base_dir = project_root / DATASET_CONFIGS[dataset_name]["spell_eval_dir"]

    return base_dir / strategy["data_file"]


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    output_suffix = f"smoke_max{args.max_examples}" if args.max_examples > 0 else None
    metrics_dir = (
        Path(args.metrics_dir)
        if args.metrics_dir
        else project_root / "results/metrics/transformer_spell"
    )
    plot_dir = (
        Path(args.plot_dir)
        if args.plot_dir
        else project_root / "results/plots/transformer_spell"
    )

    if output_suffix and args.metrics_dir is None:
        metrics_dir = metrics_dir / output_suffix
    if output_suffix and args.plot_dir is None:
        plot_dir = plot_dir / output_suffix

    datasets = selected_items(args.dataset, DATASET_CONFIGS.keys())
    strategies = selected_items(args.strategy, STRATEGY_CONFIGS.keys())
    all_results = []

    for dataset_name in datasets:
        predictor = build_predictor(args, dataset_name)

        dataset_results = []
        for strategy_name in strategies:
            config = strategy_config(strategy_name, args)
            data_path = spell_eval_path(args, dataset_name, strategy_name)
            examples = load_jsonl(data_path, max_examples=args.max_examples)

            print()
            print(f"Dataset       : {dataset_name}")
            print(f"Strategy      : {strategy_name.upper()}")
            print(f"Description   : {config['description']}")
            print(f"Spell data    : {data_path}")
            print(f"Examples read : {len(examples):,}")
            print(
                f"Corrector     : max_edit_dist={config['max_edit_dist']}, "
                f"weight_mode={config['weight_mode']}, "
                f"lambda_edit={args.lambda_edit}, mu_freq={args.mu_freq}"
            )
            print(f"Prefix range  : {args.start_chars_typed}..len(corrupted)")

            corrector = SpellCorrector(
                predictor,
                max_edit_dist=config["max_edit_dist"],
                lambda_edit=args.lambda_edit,
                mu_freq=args.mu_freq,
                weight_mode=config["weight_mode"],
            )

            result = evaluate_dataset_strategy(
                corrector=corrector,
                examples=examples,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                start_chars_typed=args.start_chars_typed,
            )
            result["configuration"] = {
                "model": "transformer",
                "spell_data_path": str(data_path),
                "max_edit_dist": config["max_edit_dist"],
                "weight_mode": config["weight_mode"],
                "lambda_edit": args.lambda_edit,
                "mu_freq": args.mu_freq,
                "top_k_values": TOP_K_VALUES,
                "main_top_k_for_breakdown_plots": MAIN_TOP_K,
                "max_examples": args.max_examples,
                "start_chars_typed": args.start_chars_typed,
            }

            print_run_summary(result)
            result_path, examples_path = save_run_outputs(result, metrics_dir)
            plot_run(result, plot_dir)
            print(f"Metrics saved : {result_path}")
            print(f"Examples saved: {examples_path}")

            all_results.append(result)
            dataset_results.append(result)

        plot_dataset_comparison(dataset_name, dataset_results, plot_dir)

    summary_paths = write_summary_outputs(all_results, metrics_dir)
    print()
    print("Combined outputs saved:")
    for path in summary_paths:
        print(f"  {path}")
    print(f"Plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()
