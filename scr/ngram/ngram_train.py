import argparse
from pathlib import Path
from ngram_model import NGramModel


def load_preprocessed_sentences(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"cant find file bro: {path}")

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


def train_ngram_model(train_path, save_path, max_n_gram, min_count):
    tokenized_sentences = load_preprocessed_sentences(train_path)

    print("Loaded preprocessed training data")
    print("Number of sentences:", len(tokenized_sentences))

    model = NGramModel(max_n_gram=max_n_gram,min_count=min_count,
    )

    model.train(tokenized_sentences)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(save_path)

    print("Training finished")
    print("Model saved to:", save_path)
    print(model)
    print("\n")
    print()

    return model


def test_model(model):
    examples = [
        "i want to eat p",
        "the dog is",
        "once upon a",
        "she went to",
    ]

    for text in examples:
        context, prefix = model.parse_text_input(text)

        print()
        print("Input:", text)
        print("Context:", context)
        print("Prefix:", prefix)

        print("Unigram:", model.predict(context, prefix, n=1, top_k=5))
        print("Bigram:", model.predict(context, prefix, n=2, top_k=5))
        print("Trigram:", model.predict(context, prefix, n=3, top_k=5))

        if model.max_n_gram >= 4:
            print("4-gram:", model.predict(context, prefix, n=4, top_k=5))

        print("Interpolated:", model.predict_interpolated(context, prefix, top_k=5))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        default="scr/data/tiny_stories/tinystories_train.txt"
    )

    parser.add_argument(
        "--save_path",
        default="models/ngram/ngram_model.pkl",
    )

    parser.add_argument(
        "--max_n_gram",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )

    args = parser.parse_args()

    model = train_ngram_model(train_path=args.train_path, save_path=args.save_path, max_n_gram=args.max_n_gram, 
                              min_count=args.min_count,
    )

    if args.test:
        test_model(model)


if __name__ == "__main__":
    main()


