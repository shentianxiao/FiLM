import os
from argparse import ArgumentParser

from utils.io import load_id, write_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    if "roberta" in args.dir:
        mask_id = 50264
    elif "gpt2" in args.dir:
        mask_id = 50257
    else:
        raise ValueError(f"Unknown tokenizer for {args.dir}")

    os.makedirs(f"{args.dir}/mask", exist_ok=True)
    for split in ["valid", "test"]:
        data = load_id(f"{args.dir}/{split}.id")
        masked_data = [[mask_id] * len(x) for x in data]
        write_id(masked_data, f"{args.dir}/mask/{split}.mask.all.id")


if __name__ == "__main__":
    main()
