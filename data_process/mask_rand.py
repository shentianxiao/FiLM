import os
from argparse import ArgumentParser
import random

from utils.io import load_id, write_id


def mask_span(data, mask_id, min_spans, max_spans):
    masked_data, infill_data = [], []
    for x in data:
        n = len(x)
        m = random.randint(min_spans, max_spans) * 2
        if m > n:
            continue
        idx = sorted(random.sample(range(n), m))
        y = x.copy()
        for i in range(0, m, 2):
            y[idx[i]: idx[i + 1]] = [mask_id] * (idx[i + 1] - idx[i])
        masked_data.append(y)
        infill_data.append(x)
    return masked_data, infill_data


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--max_spans", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    random.seed(args.seed)

    if "roberta" in args.dir:
        mask_id = 50264
    elif "gpt2" in args.dir:
        mask_id = 50257
    else:
        raise ValueError(f"Unknown tokenizer for {args.dir}")

    os.makedirs(f"{args.dir}/mask", exist_ok=True)
    for split in ["valid", "test"]:
        data = load_id(f"{args.dir}/{split}.id")

        masked_data, infill_data = mask_span(data, mask_id, 1, args.max_spans)
        write_id(masked_data, f"{args.dir}/mask/{split}.mask.span1-{args.max_spans}.id")
        write_id(infill_data, f"{args.dir}/mask/{split}.infill.span1-{args.max_spans}.id")

        for k in range(1, args.max_spans + 1):
            masked_data, infill_data = mask_span(data, mask_id, k, k)
            write_id(masked_data, f"{args.dir}/mask/{split}.mask.span{k}.id")
            write_id(infill_data, f"{args.dir}/mask/{split}.infill.span{k}.id")


if __name__ == "__main__":
    main()
