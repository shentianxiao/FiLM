import os
from argparse import ArgumentParser
import random

from .get_data import load_rocstories_sentence
from utils.io import write_id
from utils.data import get_tokenizer, tokenize


def mask_sent(sents, n, mask_id):
    if n == 1:
        mask = [random.randrange(5)]
    elif n == 2:
        mask = [1, 3]
    elif n == 3:
        mask = [0, 2, 4]
    else:
        raise ValueError

    sents = [[mask_id] * len(sent) if i in mask else sent for i, sent in enumerate(sents)]
    return [x for sent in sents for x in sent]


def main():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer", required=True, choices=["roberta", "gpt2"])
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = load_rocstories_sentence()
    tokenizer = get_tokenizer(args.tokenizer + "-large")
    mask_id = tokenizer.mask_token_id

    dir = f"data/roc_stories/{args.tokenizer}/mask"
    os.makedirs(dir, exist_ok=True)

    for split, data in dataset.items():
        if split == "train":
            continue

        data = [[tokenize(tokenizer, sent) for sent in sents] for sents in data]
        infill_data = [[x for sent in sents for x in sent] for sents in data]

        masked_data = [mask_sent(sents, 1, mask_id) for sents in data]
        write_id(masked_data, f"{dir}/{split}.mask.span1.id")
        write_id(infill_data, f"{dir}/{split}.infill.span1.id")


if __name__ == "__main__":
    main()
