from argparse import ArgumentParser

from utils.io import load_id, write_text
from utils.data import get_tokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--tokenizer", required=True, choices=["roberta", "gpt2"])
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer + "-large")
    ids = load_id(args.file)
    texts = tokenizer.batch_decode(ids, skip_special_tokens=False)
    path = args.file.replace(".id", ".json")
    write_text(texts, path)


if __name__ == "__main__":
    main()
