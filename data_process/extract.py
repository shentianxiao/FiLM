from argparse import ArgumentParser

from utils.io import load_id, write_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--infill", required=True)
    args = parser.parse_args()

    if "roberta" in args.mask:
        mask_id = 50264
    elif "gpt2" in args.mask:
        mask_id = 50257
    else:
        raise ValueError(f"Unknown tokenizer for {args.mask}")

    masked_data = load_id(args.mask)
    infill_data = load_id(args.infill)
    fill_data = []
    for x, y in zip(masked_data, infill_data):
        i = 0
        while i < len(x):
            if x[i] == mask_id:
                j = i
                while j < len(x) and x[j] == mask_id:
                    j += 1
                fill_data.append(y[i:j])
                i = j
            else:
                i += 1
    write_id(fill_data, args.infill.replace("infill", "fill"))


if __name__ == "__main__":
    main()