import os
from argparse import ArgumentParser
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from utils.io import load_text, write_id
from utils.data import MAX_LEN, get_tokenizer, tokenize


def sentence_split(tokenizer, text):
    texts = []
    sents = sent_tokenize(text)
    for sent in sents:
        if texts and len(tokenize(tokenizer, texts[-1] + " " + sent)) <= MAX_LEN:
            texts[-1] += " " + sent
        else:
            texts.append(sent)
    return texts


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--tokenizer", required=True, choices=["roberta", "gpt2"])
    parser.add_argument("--sentence_split", action="store_true")
    args = parser.parse_args()

    output_dir = f"{args.dir}/{args.tokenizer}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.tokenizer + "-large")

    for split in ["train", "valid", "test"]:
        texts = load_text(f"{args.dir}/{split}.json")

        if args.sentence_split:
            texts_split = []
            for text in tqdm(texts, desc=f"{split} sentence splitting"):
                texts_split += sentence_split(tokenizer, text)
            texts = texts_split

        ids = [tokenize(tokenizer, text) for text in tqdm(texts, desc=f"{split} tokenizing")]
        n = len(ids)
        ids = [id for id in ids if len(id) <= MAX_LEN]
        m = len(ids)
        print(f"{n-m}/{n} instances that exceed max length {MAX_LEN} are discarded")
        write_id(ids, f"{output_dir}/{split}.id")


if __name__ == "__main__":
    main()
