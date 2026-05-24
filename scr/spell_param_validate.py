"""
Validate spell-correction scoring parameters on a small corrupted validation set.

This script samples clean sentences from the validation split, corrupts only the
final word, and grid-searches:

    lambda_edit
    mu_freq

The language-model parameters are not tuned here. For the N-gram model, the
script uses the already saved best interpolation lambdas by default.

Examples:
    # Default: N-gram, TinyStories, S2, 100 validation examples
    python scr/spell_param_validate.py

    # WikiText-2, S3, custom grid
    python scr/spell_param_validate.py --dataset wikitext2 --strategy s3 \
        --lambda_edit_grid 0.25,0.5,0.75,1.0 --mu_freq_grid 0,0.05,0.1,0.2

    # Transformer version. This can be slow.
    python scr/spell_param_validate.py --model transformer --dataset tinystories --strategy s2 --device auto

Outputs:
    results/metrics/spell_param_validation/
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
NGRAM_DIR = CURRENT_DIR / "ngram"
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(NGRAM_DIR))

from ngram_model import NGramModel
from spell_corrector import SpellCorrector, damerau_levenshtein


TOP_K_VALUES = [1, 2, 3, 4]
MAX_TOP_K = max(TOP_K_VALUES)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
STRIP_CHARS = ".,!?;:\"'()[]{}@-"
EDIT_TYPES = ("insertion", "deletion", "substitution", "transposition")

DATASET_CONFIGS = {
    "tinystories": {
        "validation_text": "scr/data/tiny_stories/tinystories_val.txt",
        "ngram_model_candidates": [
            "models/ngram/Tiny_stories_ngram_model.pkl",
            "models/ngram/tiny_stories_ngram_model.pkl",
            "models/ngram/tinystories_ngram_model.pkl",
        ],
        "lambda_candidates": [
            "results/metrics/best_ngram_lambdas.json",
        ],
        "transformer_model_dir": "models/transformer/tinystories",
        "transformer_train_text": "scr/data/tiny_stories_transformer/tinystories_transformer_train.txt",
    },
    "wikitext2": {
        "validation_text": "scr/data/wikitext_2/wikitext2_val.txt",
        "ngram_model_candidates": [
            "models/ngram/Wikitext2_ngram_model.pkl",
            "models/ngram/wikitext2_ngram_model.pkl",
        ],
        "lambda_candidates": [
            "results/metrics/best_ngram_lambdasWikitext2.json",
            "results/metrics/best_ngram_lambdas_wikitext2.json",
        ],
        "transformer_model_dir": "models/transformer/wikitext2",
        "transformer_train_text": "scr/data/wikitext_2_transformer/wikitext2_transformer_train.txt",
    },
}

STRATEGY_CONFIGS = {
    "s1": {
        "edit_distance": 1,
        "max_edit_dist": 1,
        "weight_mode": "uniform",
    },
    "s2": {
        "edit_distance": 2,
        "max_edit_dist": 2,
        "weight_mode": "uniform",
    },
    "s3": {
        "edit_distance": 2,
        "max_edit_dist": 2,
        "weight_mode": "operation",
    },
}


class NGramSpellPredictor:
    """Adapter that makes NGramModel use selected interpolation lambdas."""

    def __init__(self, model, lambdas):
        self.model = model
        self.lambdas = lambdas
        self.word_counts = model.word_counts

        special_tokens = {model.start_token, model.end_token, model.unk_token}
        if getattr(model, "non_special_vocab", None):
            self.word_vocab = list(model.non_special_vocab)
        else:
            self.word_vocab = [word for word in model.vocab if word not in special_tokens]
        self.non_special_vocab = self.word_vocab
        self.word_vocab_set = set(self.word_vocab)

    def parse_text_input(self, text):
        return self.model.parse_text_input(text)

    def interpolated_probability(self, context, word):
        return self.model.interpolated_probability(
            context=context,
            word=word,
            lambdas=self.lambdas,
        )

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        return self.model.predict_interpolated(
            context=context,
            prefix=prefix,
            top_k=top_k,
            lambdas=self.lambdas if lambdas is None else lambdas,
            include_scores=include_scores,
        )

    def predict(self, context, prefix="", top_k=5, include_scores=True):
        return self.predict_interpolated(
            context=context,
            prefix=prefix,
            top_k=top_k,
            include_scores=include_scores,
        )


def resolve_existing_path(project_root, candidates):
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    return project_root / candidates[0]


def clean_token(token):
    return token.lower().strip(STRIP_CHARS)


def load_validation_sample(path, rng, sample_pool_size):
    sentence_sample = []
    vocab = set()
    candidate_count = 0

    with path.open("r", encoding="utf-8") as f:
        for sentence_id, line in enumerate(f):
            tokens = [clean_token(token) for token in line.strip().split()]
            tokens = [token for token in tokens if token]

            for token in tokens:
                if token.isalpha():
                    vocab.add(token)

            if len(tokens) < 2:
                continue
            if any(not token.isalpha() for token in tokens):
                continue
            if len(tokens[-1]) <= 1:
                continue

            candidate = {
                "source_sentence_id": sentence_id,
                "tokens": tokens,
            }
            candidate_count += 1

            if len(sentence_sample) < sample_pool_size:
                sentence_sample.append(candidate)
            else:
                replace_idx = rng.randrange(candidate_count)
                if replace_idx < sample_pool_size:
                    sentence_sample[replace_idx] = candidate

    return sentence_sample, vocab, candidate_count


def apply_one_edit(word, rng, edit_type):
    chars = list(word)

    if edit_type == "insertion":
        idx = rng.randrange(len(chars) + 1)
        chars.insert(idx, rng.choice(ALPHABET))
        return "".join(chars)

    if edit_type == "deletion":
        if len(chars) <= 2:
            return None
        idx = rng.randrange(len(chars))
        del chars[idx]
        return "".join(chars)

    if edit_type == "substitution":
        idx = rng.randrange(len(chars))
        old_char = chars[idx]
        choices = [char for char in ALPHABET if char != old_char]
        chars[idx] = rng.choice(choices)
        return "".join(chars)

    if edit_type == "transposition":
        possible = [i for i in range(len(chars) - 1) if chars[i] != chars[i + 1]]
        if not possible:
            return None
        idx = rng.choice(possible)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    raise ValueError(f"Unknown edit type: {edit_type}")


def corrupt_word(target, edit_distance, vocab, rng, max_attempts=200):
    for _ in range(max_attempts):
        corrupted = target
        operations = []

        for _ in range(edit_distance):
            edit_type = rng.choice(EDIT_TYPES)
            next_corrupted = apply_one_edit(corrupted, rng, edit_type)
            if not next_corrupted or next_corrupted == corrupted:
                break

            corrupted = next_corrupted
            operations.append(edit_type)

        if len(operations) != edit_distance:
            continue
        if corrupted == target or len(corrupted) <= 1:
            continue
        if corrupted in vocab:
            continue
        if int(damerau_levenshtein(corrupted, target)) != edit_distance:
            continue

        return corrupted, operations

    return None, None


def build_validation_examples(source_path, num_examples, seed, edit_distance, sample_pool_size):
    sample_rng = random.Random(seed)
    corrupt_rng = random.Random(seed + 1)
    candidates, vocab, candidate_count = load_validation_sample(
        source_path,
        sample_rng,
        sample_pool_size,
    )
    corrupt_rng.shuffle(candidates)

    examples = []

    for candidate in candidates:
        tokens = candidate["tokens"]
        target = tokens[-1]
        context = tokens[:-1]
        corrupted, operations = corrupt_word(
            target=target,
            edit_distance=edit_distance,
            vocab=vocab,
            rng=corrupt_rng,
        )

        if not corrupted:
            continue

        examples.append(
            {
                "source_sentence_id": candidate["source_sentence_id"],
                "word_index": len(tokens) - 1,
                "context": context,
                "target": target,
                "corrupted": corrupted,
                "edit_distance": edit_distance,
                "edit_operations": operations,
                "edit_type": "+".join(operations),
                "display_sentence": " ".join(context + [f"{corrupted} <{target}>"]),
            }
        )

        if len(examples) >= num_examples:
            break

    if len(examples) < num_examples:
        raise ValueError(
            f"Only generated {len(examples)} validation examples; requested {num_examples}. "
            f"Candidate sentences found: {candidate_count}. Try increasing --sample_pool_size."
        )

    return examples, candidate_count


def normalize_lambdas(raw_lambdas, max_n_gram):
    lambdas = {int(n): float(value) for n, value in raw_lambdas.items()}

    for n in range(1, max_n_gram + 1):
        lambdas.setdefault(n, 0.0)

    total = sum(lambdas.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"N-gram lambdas must sum to 1. Got {total}.")

    return {n: lambdas[n] for n in range(1, max_n_gram + 1)}


def load_lambdas(path, max_n_gram):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_lambdas = (
        data.get("best_lambdas")
        or data.get("lambdas")
        or data.get("best_validation_metrics", {}).get("lambdas")
        or data.get("tuning_best_metrics", {}).get("lambdas")
    )

    if raw_lambdas is None:
        raise ValueError(f"Could not find lambdas in {path}")

    return normalize_lambdas(raw_lambdas, max_n_gram=max_n_gram)


def parse_grid(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_lambdas_string(lambda_string, max_n_gram):
    values = [float(value.strip()) for value in lambda_string.split(",")]
    if len(values) != max_n_gram:
        raise ValueError(f"Expected {max_n_gram} lambda values, got {len(values)}.")
    return normalize_lambdas(
        {n: value for n, value in enumerate(values, start=1)},
        max_n_gram=max_n_gram,
    )


def build_ngram_predictor(args, project_root, dataset_config):
    model_path = (
        Path(args.ngram_model_path)
        if args.ngram_model_path
        else resolve_existing_path(project_root, dataset_config["ngram_model_candidates"])
    )
    model = NGramModel.load(model_path)

    if args.ngram_lambdas:
        lambdas = parse_lambdas_string(args.ngram_lambdas, model.max_n_gram)
        lambda_source = "command_line"
    else:
        lambda_path = (
            Path(args.ngram_lambda_path)
            if args.ngram_lambda_path
            else resolve_existing_path(project_root, dataset_config["lambda_candidates"])
        )
        lambdas = load_lambdas(lambda_path, max_n_gram=model.max_n_gram)
        lambda_source = str(lambda_path)

    return NGramSpellPredictor(model, lambdas), str(model_path), lambda_source, lambdas


def build_transformer_predictor(args, project_root, dataset_config):
    from transformer_model import TransformerPredictor

    model_dir = project_root / dataset_config["transformer_model_dir"]
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

    predictor = TransformerPredictor(
        str(checkpoint_path),
        str(tokenizer_path),
        device=args.device,
    )

    ngram_path = resolve_existing_path(project_root, dataset_config["ngram_model_candidates"])
    if ngram_path.exists():
        predictor.load_word_vocab_from_ngram(str(ngram_path))
        vocab_source = str(ngram_path)
    else:
        train_path = project_root / dataset_config["transformer_train_text"]
        predictor.build_word_vocab_from_text(str(train_path))
        vocab_source = str(train_path)

    return predictor, str(checkpoint_path), vocab_source, None


def get_rank(suggestions, target_word):
    if suggestions and not isinstance(suggestions[0], str):
        suggestions = [word for word, _ in suggestions]

    for rank, word in enumerate(suggestions, start=1):
        if word == target_word:
            return rank
    return None


def evaluate_examples(corrector, examples, start_chars_typed):
    vocab_set = set(corrector.word_vocab)
    total_examples = 0
    skipped_oov = 0
    total_characters = 0
    correct_by_k = {k: 0 for k in TOP_K_VALUES}
    saved_by_k = {k: 0 for k in TOP_K_VALUES}

    for example in examples:
        target = example["target"]
        corrupted = example["corrupted"]
        context = example["context"]

        if target not in vocab_set:
            skipped_oov += 1
            continue

        total_examples += 1
        total_characters += len(target)

        found = {k: False for k in TOP_K_VALUES}
        chars_found = {k: len(target) for k in TOP_K_VALUES}
        first_prefix_len = min(start_chars_typed, len(corrupted))

        for chars_typed in range(first_prefix_len, len(corrupted) + 1):
            if all(found.values()):
                break

            suggestions = corrector.predict(
                context=context,
                prefix=corrupted[:chars_typed],
                top_k=MAX_TOP_K,
                include_scores=False,
                use_lm=True,
            )
            rank = get_rank(suggestions, target)

            if rank is None:
                continue

            for k in TOP_K_VALUES:
                if not found[k] and rank <= k:
                    found[k] = True
                    chars_found[k] = chars_typed

        for k in TOP_K_VALUES:
            if found[k]:
                correct_by_k[k] += 1
            saved_by_k[k] += max(0, len(target) - chars_found[k])

    results_by_top_k = {}
    for k in TOP_K_VALUES:
        results_by_top_k[str(k)] = {
            "top_k": k,
            "total_examples": total_examples,
            "total_characters": total_characters,
            "top_k_correct_words": correct_by_k[k],
            "top_k_accuracy": correct_by_k[k] / total_examples if total_examples else 0.0,
            "saved_keystrokes": saved_by_k[k],
            "saved_keystroke_ratio": saved_by_k[k] / total_characters if total_characters else 0.0,
        }

    return {
        "loaded_examples": len(examples),
        "evaluated_examples": total_examples,
        "skipped_oov_examples": skipped_oov,
        "results_by_top_k": results_by_top_k,
    }


def objective_value(metrics, top_k, objective):
    return metrics["results_by_top_k"][str(top_k)][objective]


def run_grid_search(args, corrector_base, examples, strategy_config):
    lambda_edit_values = parse_grid(args.lambda_edit_grid)
    mu_freq_values = parse_grid(args.mu_freq_grid)
    rows = []

    for lambda_edit in tqdm(lambda_edit_values, desc="lambda_edit grid"):
        for mu_freq in mu_freq_values:
            corrector = SpellCorrector(
                corrector_base,
                max_edit_dist=strategy_config["max_edit_dist"],
                lambda_edit=lambda_edit,
                mu_freq=mu_freq,
                weight_mode=strategy_config["weight_mode"],
            )
            metrics = evaluate_examples(
                corrector=corrector,
                examples=examples,
                start_chars_typed=args.start_chars_typed,
            )
            row = {
                "lambda_edit": lambda_edit,
                "mu_freq": mu_freq,
                "objective_top_k": args.objective_top_k,
                "objective": args.objective,
                "objective_value": objective_value(
                    metrics,
                    top_k=args.objective_top_k,
                    objective=args.objective,
                ),
                **metrics,
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["objective_value"],
            row["results_by_top_k"][str(args.objective_top_k)]["top_k_accuracy"],
            row["results_by_top_k"][str(args.objective_top_k)]["saved_keystroke_ratio"],
        ),
        reverse=True,
    )
    return rows


def save_outputs(args, project_root, examples, rows, metadata):
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results/metrics/spell_param_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.model}_{args.dataset}_{args.strategy}_{args.num_examples}examples"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    examples_path = output_dir / f"{stem}_examples.jsonl"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "best_result": rows[0],
                "all_results": rows,
            },
            f,
            indent=4,
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "lambda_edit",
            "mu_freq",
            "objective_top_k",
            "objective",
            "objective_value",
            "evaluated_examples",
            "skipped_oov_examples",
        ]
        for k in TOP_K_VALUES:
            fieldnames.extend(
                [
                    f"top_{k}_accuracy",
                    f"top_{k}_saved_keystroke_ratio",
                    f"top_{k}_saved_keystrokes",
                ]
            )

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            csv_row = {
                "rank": index,
                "lambda_edit": row["lambda_edit"],
                "mu_freq": row["mu_freq"],
                "objective_top_k": row["objective_top_k"],
                "objective": row["objective"],
                "objective_value": row["objective_value"],
                "evaluated_examples": row["evaluated_examples"],
                "skipped_oov_examples": row["skipped_oov_examples"],
            }
            for k in TOP_K_VALUES:
                metrics = row["results_by_top_k"][str(k)]
                csv_row[f"top_{k}_accuracy"] = metrics["top_k_accuracy"]
                csv_row[f"top_{k}_saved_keystroke_ratio"] = metrics["saved_keystroke_ratio"]
                csv_row[f"top_{k}_saved_keystrokes"] = metrics["saved_keystrokes"]
            writer.writerow(csv_row)

    with examples_path.open("w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=True)
            f.write("\n")

    return json_path, csv_path, examples_path


def parse_args():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Validate spell-correction lambda_edit and mu_freq on corrupted validation examples."
    )
    parser.add_argument("--model", choices=["ngram", "transformer"], default="ngram")
    parser.add_argument("--dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--strategy", choices=["s1", "s2", "s3"], default="s2")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample_pool_size", type=int, default=5000)
    parser.add_argument("--lambda_edit_grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--mu_freq_grid", default="0.0,0.05,0.1,0.2")
    parser.add_argument(
        "--objective",
        choices=["saved_keystroke_ratio", "top_k_accuracy"],
        default="saved_keystroke_ratio",
    )
    parser.add_argument("--objective_top_k", type=int, choices=TOP_K_VALUES, default=3)
    parser.add_argument("--start_chars_typed", type=int, default=1)
    parser.add_argument("--project_root", default=str(project_root))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ngram_model_path", default=None)
    parser.add_argument("--ngram_lambda_path", default=None)
    parser.add_argument("--ngram_lambdas", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    dataset_config = DATASET_CONFIGS[args.dataset]
    strategy_config = STRATEGY_CONFIGS[args.strategy]
    validation_path = project_root / dataset_config["validation_text"]

    print("Spell parameter validation")
    print(f"Model           : {args.model}")
    print(f"Dataset         : {args.dataset}")
    print(f"Strategy        : {args.strategy}")
    print(f"Validation text : {validation_path}")
    print(f"Examples        : {args.num_examples}")
    print(f"Objective       : top-{args.objective_top_k} {args.objective}")
    print(f"lambda_edit grid: {args.lambda_edit_grid}")
    print(f"mu_freq grid    : {args.mu_freq_grid}")

    examples, candidate_count = build_validation_examples(
        source_path=validation_path,
        num_examples=args.num_examples,
        seed=args.seed,
        edit_distance=strategy_config["edit_distance"],
        sample_pool_size=args.sample_pool_size,
    )

    if args.model == "ngram":
        predictor, model_path, aux_source, lambdas = build_ngram_predictor(
            args,
            project_root,
            dataset_config,
        )
    else:
        predictor, model_path, aux_source, lambdas = build_transformer_predictor(
            args,
            project_root,
            dataset_config,
        )

    rows = run_grid_search(
        args=args,
        corrector_base=predictor,
        examples=examples,
        strategy_config=strategy_config,
    )

    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "validation_text": str(validation_path),
        "candidate_sentences": candidate_count,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "sample_pool_size": args.sample_pool_size,
        "edit_distance": strategy_config["edit_distance"],
        "max_edit_dist": strategy_config["max_edit_dist"],
        "weight_mode": strategy_config["weight_mode"],
        "model_path": model_path,
        "aux_source": aux_source,
        "ngram_lambdas": lambdas,
        "lambda_edit_grid": parse_grid(args.lambda_edit_grid),
        "mu_freq_grid": parse_grid(args.mu_freq_grid),
        "objective": args.objective,
        "objective_top_k": args.objective_top_k,
        "start_chars_typed": args.start_chars_typed,
    }

    json_path, csv_path, examples_path = save_outputs(
        args=args,
        project_root=project_root,
        examples=examples,
        rows=rows,
        metadata=metadata,
    )

    best = rows[0]
    best_metrics = best["results_by_top_k"][str(args.objective_top_k)]

    print()
    print("=" * 72)
    print("Best spell parameters")
    print("=" * 72)
    print(f"lambda_edit: {best['lambda_edit']}")
    print(f"mu_freq    : {best['mu_freq']}")
    print(f"Objective  : {best['objective_value']:.4f}")
    print(f"Top-{args.objective_top_k} accuracy: {best_metrics['top_k_accuracy']:.4f}")
    print(f"Top-{args.objective_top_k} saved ratio: {best_metrics['saved_keystroke_ratio']:.4f}")
    print(f"Evaluated examples: {best['evaluated_examples']}")
    print(f"Skipped OOV examples: {best['skipped_oov_examples']}")
    print()
    print(f"JSON saved    : {json_path}")
    print(f"CSV saved     : {csv_path}")
    print(f"Examples saved: {examples_path}")


if __name__ == "__main__":
    main()
