import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from tokenizer import TinyStoriesTokenizer
from transformer_model import Config, TinyStoriesLM


def train_or_load_tokenizer(train_text_path, tokenizer_path, vocab_size, max_tokenizer_lines):
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        print(f"Loading existing tokenizer from {tokenizer_path}")
        return TinyStoriesTokenizer.load(str(tokenizer_path))

    print(f"Training BPE tokenizer (vocab_size={vocab_size}, max_lines={max_tokenizer_lines or 'all'})...")
    tokenizer = TinyStoriesTokenizer(vocab_size=vocab_size)
    tokenizer.train(str(train_text_path), max_lines=max_tokenizer_lines)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer


def tokenize_to_bin(text_path, tokenizer, bin_path):
    bin_path = Path(bin_path)
    if bin_path.exists():
        print(f"Binary already exists, skipping: {bin_path}")
        return

    print(f"Tokenizing {text_path} → {bin_path} ...")
    all_ids = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, ids = tokenizer.tokenize(line)
            all_ids.extend(ids)

    data = np.array(all_ids, dtype=np.uint16)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(bin_path))
    print(f"  {len(all_ids):,} tokens written.")


class TokenDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class ResumableRandomSampler(Sampler):
    def __init__(self, data_source, start_index=0, seed=42):
        self.data_source = data_source
        self.start_index = start_index
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.indices = torch.randperm(len(data_source), generator=self.generator).tolist()

    def __iter__(self):
        return iter(self.indices[self.start_index:])

    def __len__(self):
        return len(self.indices) - self.start_index


def save_checkpoint(model, optimizer, config, epoch, iteration, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            "config": config.__dict__,
        },
        str(path),
    )
    print(f"Checkpoint saved: {path}")


def train(args):
    train_text = Path(args.train_text)
    val_text = Path(args.val_text)
    tokenizer_path = Path(args.tokenizer_path)
    train_bin = Path(args.train_bin)
    val_bin = Path(args.val_bin)
    checkpoint_path = Path(args.checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    tokenizer = train_or_load_tokenizer(
        train_text, tokenizer_path,
        vocab_size=args.vocab_size,
        max_tokenizer_lines=args.max_tokenizer_lines,
    )

    tokenize_to_bin(train_text, tokenizer, train_bin)
    tokenize_to_bin(val_text, tokenizer, val_bin)

    resume = checkpoint_path.exists()
    if resume:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        config = Config(**checkpoint["config"])
        model = TinyStoriesLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed (epoch {checkpoint['epoch']}, iter {checkpoint['iteration']})")
    else:
        config = Config(
            vocab_size=tokenizer.vocab_size,
            number_of_transformer_blocks=args.n_blocks,
            number_of_attention_heads=args.n_heads,
            vector_dim=args.vector_dim,
            block_size=args.block_size,
            dropout_prob=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            no_of_epochs=args.epochs,
        )
        model = TinyStoriesLM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    criterion = nn.CrossEntropyLoss()

    train_dataset = TokenDataset(train_bin, config.block_size)
    val_dataset = TokenDataset(val_bin, config.block_size)
    val_indices = list(range(0, len(val_dataset), config.block_size))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, config.batch_size, pin_memory=(device == "cuda"), num_workers=2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Train tokens: {len(train_dataset):,}  Val tokens: {len(val_dataset):,}")

    start_epoch = checkpoint["epoch"] if resume else 0
    start_iter = checkpoint["iteration"] if resume else 0

    model.train()
    running_loss = 0.0
    print(f"\n{datetime.now().strftime('%X')} Training starts")

    for epoch in range(start_epoch, config.no_of_epochs):
        iteration = start_iter if epoch == start_epoch else 0
        sampler = ResumableRandomSampler(
            train_dataset, start_index=iteration * config.batch_size, seed=42 + epoch
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=2
        )

        for x, y in train_loader:
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
                    val_loss = sum(
                        criterion(
                            model(vx.to(device)).reshape(-1, config.vocab_size),
                            vy.to(device).reshape(-1),
                        ).item()
                        for vx, vy in val_loader
                    )
                print(
                    f"{datetime.now().strftime('%X')} Epoch {epoch+1}, iter {iteration}: "
                    f"loss={running_loss/100:.4f}, val_loss={val_loss/len(val_loader):.4f}"
                )
                model.train()
                running_loss = 0.0

            if iteration % 10000 == 0:
                save_checkpoint(model, optimizer, config, epoch, iteration, checkpoint_path)

    save_checkpoint(model, optimizer, config, config.no_of_epochs - 1, iteration, checkpoint_path)
    print(f"\nTraining complete. Checkpoint: {checkpoint_path}")


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--project_root", default=str(project_root))
    parser.add_argument("--train_text", default=None)
    parser.add_argument("--val_text", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--train_bin", default=None)
    parser.add_argument("--val_bin", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--max_tokenizer_lines", type=int, default=None)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--vector_dim", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    if args.dataset == "wikitext2":
        data_dir = project_root / "scr/data/wikitext_2_transformer"
        model_dir = project_root / "models/transformer/wikitext2"
        prefix = "wikitext2"
    else:
        data_dir = project_root / "scr/data/tiny_stories_transformer"
        model_dir = project_root / "models/transformer/tinystories"
        prefix = "tinystories"

    args.train_text = args.train_text or str(data_dir / f"{prefix}_transformer_train.txt")
    args.val_text = args.val_text or str(data_dir / f"{prefix}_transformer_val.txt")
    args.tokenizer_path = args.tokenizer_path or str(model_dir / "tokenizer.json")
    args.train_bin = args.train_bin or str(model_dir / "train.bin")
    args.val_bin = args.val_bin or str(model_dir / "val.bin")
    args.checkpoint_path = args.checkpoint_path or str(model_dir / "checkpoint.pt")

    print(f"Dataset: {args.dataset}")
    print(f"Train:   {args.train_text}")
    print(f"Val:     {args.val_text}")
    print(f"Model:   {args.checkpoint_path}")
    print()

    train(args)


if __name__ == "__main__":
    main()
