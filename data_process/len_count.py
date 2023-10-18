from argparse import ArgumentParser
from collections import Counter

from utils.io import load_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    ids = load_id(args.file)
    cnt = Counter([len(id) for id in ids])
    lens = sorted(cnt.keys())

    path = args.file.replace(".id", ".len_count")
    with open(path, "w") as f:
        for l in lens:
            f.write(f"{l}\t{cnt[l]}\n")


if __name__ == "__main__":
    main()