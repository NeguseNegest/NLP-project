import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from transformer_model import TransformerPredictor

TOP_K_VALUES = [1, 2, 3, 4]
MAX_TOP_K = max(TOP_K_VALUES)


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


def get_rank(suggestions, target_word):
    for i, word in enumerate(suggestions, start=1):
        if word == target_word:
            return i
    return None


def evaluate(predictor, sentences):
    total_words = 0
    total_characters = 0
    saved_by_k = {k: 0 for k in TOP_K_VALUES}
    top_k_correct_by_k = {k: 0 for k in TOP_K_VALUES}

    for sentence in tqdm(sentences, desc="Evaluating transformer"):
        sentence = [w for w in sentence if w]

        for i, target_word in enumerate(sentence):
            context = sentence[:i]
            total_words += 1
            total_characters += len(target_word)

            if target_word not in predictor.prefix_index.get(target_word, [target_word]):
                continue

            empty_suggestions = predictor.predict(
                context, prefix="", top_k=MAX_TOP_K, include_scores=False
            )
            empty_rank = get_rank(empty_suggestions, target_word)
            if empty_rank is not None:
                for k in TOP_K_VALUES:
                    if empty_rank <= k:
                        top_k_correct_by_k[k] += 1

            found = {k: False for k in TOP_K_VALUES}
            chars_found = {k: len(target_word) for k in TOP_K_VALUES}

            for chars_typed in range(1, len(target_word) + 1):
                if all(found.values()):
                    break
                prefix = target_word[:chars_typed]
                suggestions = predictor.predict(
                    context, prefix=prefix, top_k=MAX_TOP_K, include_scores=False
                )
                rank = get_rank(suggestions, target_word)
                if rank is None:
                    continue
                for k in TOP_K_VALUES:
                    if not found[k] and rank <= k:
                        found[k] = True
                        chars_found[k] = chars_typed

            for k in TOP_K_VALUES:
                saved_by_k[k] += max(0, len(target_word) - chars_found[k])

    results = {}
    for k in TOP_K_VALUES:
        results[k] = {
            "top_k": k,
            "total_words": total_words,
            "total_characters": total_characters,
            "saved_keystrokes": saved_by_k[k],
            "saved_keystroke_ratio": saved_by_k[k] / total_characters if total_characters > 0 else 0.0,
            "top_k_correct_words": top_k_correct_by_k[k],
            "top_k_accuracy": top_k_correct_by_k[k] / total_words if total_words > 0 else 0.0,
        }
    return results


def print_results(results, dataset_name):
    for k in TOP_K_VALUES:
        m = results[k]
        print()
        print("=" * 60)
        print(f"Transformer — {dataset_name} — top-{k}")
        print("=" * 60)
        print(f"Total words           : {m['total_words']:,}")
        print(f"Total characters      : {m['total_characters']:,}")
        print(f"Saved keystrokes      : {m['saved_keystrokes']:,}")
        print(f"Saved-keystroke ratio : {m['saved_keystroke_ratio']:.4f}")
        print(f"Top-{k} accuracy      : {m['top_k_accuracy']:.4f}")


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--project_root", default=str(project_root))
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--ngram_model_path", default=None)
    parser.add_argument("--vocab_train_text", default=None)
    parser.add_argument("--max_test_sentences", type=int, default=0)
    parser.add_argument("--device", default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    model_dir = project_root / "models/transformer/tinystories"
    checkpoint_path = args.checkpoint_path or str(model_dir / "checkpoint.pt")
    tokenizer_path = args.tokenizer_path or str(model_dir / "tokenizer.json")

    if args.dataset == "wikitext2":
        default_test = project_root / "scr/data/wikitext_2_transformer/wikitext2_transformer_test.txt"
        default_results = project_root / "results/metrics/transformer_wikitext2_test_results.json"
    else:
        default_test = project_root / "scr/data/tiny_stories_transformer/tinystories_transformer_test.txt"
        default_results = project_root / "results/metrics/transformer_tinystories_test_results.json"

    test_path = args.test_path or str(default_test)
    results_path = args.results_path or str(default_results)

    print(f"Test dataset : {args.dataset}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Tokenizer    : {tokenizer_path}")
    print(f"Test data    : {test_path}")
    print()

    predictor = TransformerPredictor(checkpoint_path, tokenizer_path, device=args.device)

    if args.ngram_model_path:
        predictor.load_word_vocab_from_ngram(args.ngram_model_path)
    elif args.vocab_train_text:
        predictor.build_word_vocab_from_text(args.vocab_train_text)
    else:
        default_ngram = project_root / "models/ngram/Tiny_stories_ngram_model.pkl"
        if default_ngram.exists():
            predictor.load_word_vocab_from_ngram(str(default_ngram))
        else:
            fallback = project_root / "scr/data/tiny_stories_transformer/tinystories_transformer_train.txt"
            predictor.build_word_vocab_from_text(str(fallback))

    print(f"Word vocab size: {len(predictor.word_vocab):,}")

    max_sentences = args.max_test_sentences if args.max_test_sentences > 0 else None
    sentences = load_sentences(test_path, max_sentences)
    print(f"Test sentences: {len(sentences):,}")

    results = evaluate(predictor, sentences)
    print_results(results, args.dataset)

    output = {
        "dataset": args.dataset,
        "checkpoint_path": checkpoint_path,
        "test_path": test_path,
        "top_k_values": TOP_K_VALUES,
        "max_test_sentences": max_sentences,
        "results_by_top_k": results,
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
