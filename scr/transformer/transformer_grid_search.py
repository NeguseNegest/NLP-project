import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_log_file = None

def log(*args, **kwargs):
    print(*args, **kwargs)
    if _log_file is not None:
        print(*args, **kwargs, file=_log_file, flush=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from transformer_model import Config, TinyStoriesLM, TransformerPredictor
from transformer_train import (
    train_or_load_tokenizer,
    tokenize_to_bin,
    TokenDataset,
    ResumableRandomSampler,
    save_checkpoint,
)
from transformer_evaluate import load_sentences, evaluate

ARCH_GRID = [
    {"name": "tiny",    "n_blocks": 1, "n_heads": 1, "vector_dim": 128},
    {"name": "small",   "n_blocks": 2, "n_heads": 2, "vector_dim": 128},
    {"name": "default", "n_blocks": 4, "n_heads": 4, "vector_dim": 256},
    {"name": "medium",  "n_blocks": 6, "n_heads": 8, "vector_dim": 512},
]

STRIDE_LEVELS = [8, 32, 64]
TOP_K_VALUES  = [1, 2, 3, 4]

def train_model(arch, stride, args, device, train_bin, val_loader, tokenizer, checkpoint_path):
    config = Config(
        vocab_size=tokenizer.vocab_size,
        number_of_transformer_blocks=arch["n_blocks"],
        number_of_attention_heads=arch["n_heads"],
        vector_dim=arch["vector_dim"],
        block_size=args.block_size,
        dropout_prob=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        no_of_epochs=1,
    )

    model = TinyStoriesLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_dataset = TokenDataset(train_bin, config.block_size, stride=stride)
    sampler = ResumableRandomSampler(train_dataset, start_index=0, seed=42)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=2
    )

    log(
        f"\n{'=' * 68}\n"
        f"  arch={arch['name']:8s}  stride={stride:>3}  |  "
        f"{arch['n_blocks']} blocks, {arch['n_heads']} heads, dim={arch['vector_dim']}  |  {n_params:,} params\n"
        f"{'=' * 68}"
    )
    log(f"{datetime.now().strftime('%X')} Training  (max {args.max_iters} iters, stride={stride})")

    model.train()
    running_loss = 0.0
    iteration = 0
    final_val_loss = None
    best_val_loss = float("inf")
    best_iteration = 0

    for x, y in train_loader:
        if iteration >= args.max_iters:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        iteration += 1

        if iteration % 100 == 0:
            model.eval()
            with torch.no_grad():
                v_loss = sum(
                    criterion(
                        model(vx.to(device)).reshape(-1, config.vocab_size),
                        vy.to(device).reshape(-1),
                    ).item()
                    for vx, vy in val_loader
                )

            final_val_loss = v_loss / len(val_loader)

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_iteration = iteration
                save_checkpoint(model, optimizer, config, 0, iteration, checkpoint_path)
                best_marker = "  [best]"
            else:
                best_marker = ""

            log(
                f"{datetime.now().strftime('%X')} iter {iteration:>6}: "
                f"loss={running_loss / 100:.4f}  val_loss={final_val_loss:.4f}"
                f"{best_marker}"
            )
            model.train()
            running_loss = 0.0

    if best_iteration == 0:
        save_checkpoint(model, optimizer, config, 0, iteration, checkpoint_path)
        best_val_loss = final_val_loss
        best_iteration = iteration

    return model, config, n_params, iteration, best_val_loss


def quick_evaluate(checkpoint_path, tokenizer_path, test_sentences, vocab_source, device):
    predictor = TransformerPredictor(str(checkpoint_path), str(tokenizer_path), device=device)

    if isinstance(vocab_source, str) and vocab_source.endswith(".pkl"):
        predictor.load_word_vocab_from_ngram(vocab_source)
    else:
        predictor.build_word_vocab_from_text(str(vocab_source))

    results = evaluate(predictor, test_sentences)
    return results


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",      choices=["tinystories", "wikitext2"], default="wikitext2")
    parser.add_argument("--project_root", default=str(project_root))
    parser.add_argument("--max_iters",    type=int,   default=10000)
    parser.add_argument("--block_size",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--vocab_size",   type=int,   default=5000)
    parser.add_argument("--max_eval_sentences", type=int, default=200,
                        help="Test sentences used for quick evaluation per run")
    parser.add_argument("--device",       default=None)

    return parser.parse_args()


def main():
    global _log_file
    args = parse_args()
    project_root = Path(args.project_root)

    if args.vocab_size is None:
        args.vocab_size = 8000 if args.dataset == "wikitext2" else 5000

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

    train_text     = data_dir  / f"{prefix}_transformer_train.txt"
    val_text       = data_dir  / f"{prefix}_transformer_val.txt"
    test_text      = data_dir  / f"{prefix}_transformer_test.txt"
    tokenizer_path = model_dir / "tokenizer.json"
    train_bin      = model_dir / "train.bin"
    val_bin        = model_dir / "val.bin"
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / f"results/metrics/{prefix}_transformer_grid_search.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "w", encoding="utf-8")
    log(f"Grid search started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log(f"Dataset : {args.dataset}")
    log(f"Device  : {device}")
    log(f"Max iter: {args.max_iters}")
    log(f"Archs   : {[a['name'] for a in ARCH_GRID]}")
    log(f"Strides : {STRIDE_LEVELS}")
    log(f"Eval on : {args.max_eval_sentences} test sentences per run")

    tokenizer = train_or_load_tokenizer(
        train_text, tokenizer_path, vocab_size=args.vocab_size, max_tokenizer_lines=None
    )
    tokenize_to_bin(train_text, tokenizer, train_bin)
    tokenize_to_bin(val_text,   tokenizer, val_bin)

    val_dataset = TokenDataset(val_bin, args.block_size, stride=args.block_size)
    val_indices = list(range(0, len(val_dataset), args.block_size))
    val_subset  = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader  = DataLoader(
        val_subset, args.batch_size,
        pin_memory=(device == "cuda"), num_workers=2,
    )

    vocab_source = str(ngram_pkl) if ngram_pkl.exists() else str(train_text)
    test_sentences = load_sentences(str(test_text), max_sentences=args.max_eval_sentences)
    log(f"\nLoaded {len(test_sentences)} test sentences for evaluation.")

    all_results = []

    for arch in ARCH_GRID:
        for stride in STRIDE_LEVELS:
            checkpoint_path = model_dir / f"{prefix}_grid_{arch['name']}_s{stride}_checkpoint.pt"

            model, config, n_params, iters, val_loss = train_model(
                arch, stride, args, device, train_bin, val_loader, tokenizer, checkpoint_path
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            log(f"{datetime.now().strftime('%X')} Running evaluation on {len(test_sentences)} sentences...")
            eval_results = quick_evaluate(
                checkpoint_path, tokenizer_path, test_sentences, vocab_source, device
            )

            row = {
                "arch":       arch["name"],
                "n_blocks":   arch["n_blocks"],
                "n_heads":    arch["n_heads"],
                "vector_dim": arch["vector_dim"],
                "stride":     stride,
                "n_params":   n_params,
                "iterations": iters,
                "val_loss":   val_loss,
                "eval":       {
                    str(k): {
                        "top_k_accuracy":       eval_results[k]["top_k_accuracy"],
                        "saved_keystroke_ratio": eval_results[k]["saved_keystroke_ratio"],
                    }
                    for k in TOP_K_VALUES
                },
                "checkpoint": str(checkpoint_path),
            }
            all_results.append(row)

            log(
                f"  val_loss={val_loss:.4f}  "
                f"top1_acc={eval_results[1]['top_k_accuracy']:.4f}  "
                f"saved_ratio={eval_results[1]['saved_keystroke_ratio']:.4f}"
            )

    log(f"\n{'=' * 100}")
    log(f"{'Grid search summary — sorted by top-1 accuracy':^100}")
    log(f"{'=' * 100}")
    header = (
        f"{'Arch':<10} {'Stride':>6} {'Blocks':>6} {'Heads':>5} {'Dim':>5} "
        f"{'Params':>10} {'ValLoss':>8} "
        f"{'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'KS-ratio':>9}"
    )
    log(header)
    log("-" * 100)
    for r in sorted(all_results, key=lambda x: -x["eval"]["1"]["top_k_accuracy"]):
        log(
            f"{r['arch']:<10} {r['stride']:>6} {r['n_blocks']:>6} {r['n_heads']:>5} {r['vector_dim']:>5} "
            f"{r['n_params']:>10,} {r['val_loss']:>8.4f} "
            f"{r['eval']['1']['top_k_accuracy']:>6.4f} "
            f"{r['eval']['2']['top_k_accuracy']:>6.4f} "
            f"{r['eval']['3']['top_k_accuracy']:>6.4f} "
            f"{r['eval']['4']['top_k_accuracy']:>6.4f} "
            f"{r['eval']['1']['saved_keystroke_ratio']:>9.4f}"
        )

    results_path = project_root / f"results/metrics/{prefix}_transformer_grid_search.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {"dataset": args.dataset, "max_iters": args.max_iters, "runs": all_results},
            f, indent=4,
        )
    log(f"\nFull results saved to {results_path}")
    log(f"Log saved to {log_path}")
    _log_file.close()


if __name__ == "__main__":
    main()
