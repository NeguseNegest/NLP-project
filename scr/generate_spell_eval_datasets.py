"""
Generate fixed misspelled-word evaluation datasets.

Creates exact edit-distance 1 and 2 JSONL files for each dataset.
"""

import argparse
import json
import random
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from spell_corrector import damerau_levenshtein


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
STRIP_CHARS = ".,!?;:\"'()[]{}@-"
EDIT_TYPES = ("insertion", "deletion", "substitution", "transposition")


DATASETS = {
    "tinystories": "scr/data/ngram_tiny_stories/tinystories_test.txt",
    "wikitext2": "scr/data/ngram_wikitext_2/wikitext2_test.txt",
    "mobile_sms": "scr/data/ngram_mobile/test_sms.txt",
}


def clean_token(token):
    return token.lower().strip(STRIP_CHARS)


def load_candidate_sample(path, rng, sample_pool_size):
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

            target = tokens[-1]
            if len(target) <= 1:
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
        if len(chars) < 2:
            return None
        possible = [i for i in range(len(chars) - 1) if chars[i] != chars[i + 1]]
        if not possible:
            return None
        idx = rng.choice(possible)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    raise ValueError(f"Unknown edit type: {edit_type}")


def corrupt_word(target, edit_distance, vocab, rng, max_attempts=200):
    if edit_distance not in {1, 2}:
        raise ValueError("Only edit distances 1 and 2 are supported.")

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


def make_display_sentence(context, corrupted, target):
    return " ".join(context + [f"{corrupted} <{target}>"])


def build_examples(dataset_name, source_path, num_examples, seed, sample_pool_size):
    sample_rng = random.Random(seed)
    corrupt_rng = random.Random(seed + 1)
    candidates, vocab, candidate_count = load_candidate_sample(
        source_path, sample_rng, sample_pool_size
    )
    rng = corrupt_rng
    rng.shuffle(candidates)

    edit1_examples = []
    edit2_examples = []

    for candidate in candidates:
        tokens = candidate["tokens"]
        context = tokens[:-1]
        target = tokens[-1]

        corrupted_1, ops_1 = corrupt_word(target, 1, vocab, rng)
        corrupted_2, ops_2 = corrupt_word(target, 2, vocab, rng)

        if not corrupted_1 or not corrupted_2:
            continue

        base = {
            "dataset": dataset_name,
            "source_sentence_id": candidate["source_sentence_id"],
            "word_index": len(tokens) - 1,
            "context": context,
            "target": target,
        }

        edit1_examples.append(
            {
                **base,
                "example_id": len(edit1_examples),
                "corrupted": corrupted_1,
                "edit_distance": 1,
                "edit_operations": ops_1,
                "edit_type": "+".join(ops_1),
                "display_sentence": make_display_sentence(context, corrupted_1, target),
            }
        )
        edit2_examples.append(
            {
                **base,
                "example_id": len(edit2_examples),
                "corrupted": corrupted_2,
                "edit_distance": 2,
                "edit_operations": ops_2,
                "edit_type": "+".join(ops_2),
                "display_sentence": make_display_sentence(context, corrupted_2, target),
            }
        )

        if len(edit1_examples) >= num_examples:
            break

    if len(edit1_examples) < num_examples:
        raise ValueError(
            f"Only generated {len(edit1_examples)} paired examples for {dataset_name}; "
            f"requested {num_examples}. Try increasing --sample_pool_size."
        )

    return edit1_examples, edit2_examples, candidate_count, len(vocab)


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=True)
            f.write("\n")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Generate fixed misspelled-word test datasets."
    )
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to generate. Defaults to all configured datasets.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sample_pool_size",
        type=int,
        default=20000,
        help="Number of valid candidate sentences to keep in memory per dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(project_root / "scr/data/data_for_spell_evaluation"),
    )
    parser.add_argument(
        "--project_root",
        default=str(project_root),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)

    summary = {}

    selected_datasets = (
        DATASETS.items()
        if args.dataset == "all"
        else [(args.dataset, DATASETS[args.dataset])]
    )

    for offset, (dataset_name, relative_path) in enumerate(selected_datasets):
        source_path = project_root / relative_path
        dataset_seed = args.seed + offset * 1000

        edit1_examples, edit2_examples, candidate_count, vocab_size = build_examples(
            dataset_name=dataset_name,
            source_path=source_path,
            num_examples=args.num_examples,
            seed=dataset_seed,
            sample_pool_size=args.sample_pool_size,
        )

        dataset_dir = output_dir / dataset_name
        edit1_path = dataset_dir / "test_edit1.jsonl"
        edit2_path = dataset_dir / "test_edit2.jsonl"

        write_jsonl(edit1_path, edit1_examples)
        write_jsonl(edit2_path, edit2_examples)

        summary[dataset_name] = {
            "source_path": str(source_path),
            "seed": dataset_seed,
            "candidate_sentences": candidate_count,
            "sample_pool_size": args.sample_pool_size,
            "vocab_size": vocab_size,
            "num_examples": args.num_examples,
            "edit1_path": str(edit1_path),
            "edit2_path": str(edit2_path),
        }

        print(
            f"{dataset_name}: wrote {args.num_examples} paired examples "
            f"to {edit1_path} and {edit2_path}"
        )

    summary_path = output_dir / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=True)

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
