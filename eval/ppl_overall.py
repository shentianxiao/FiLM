from argparse import ArgumentParser
import numpy as np

from utils.data import MAX_LEN
from utils.io import load_pair


def main():
    parser = ArgumentParser()
    parser.add_argument("--len_count", required=True)
    parser.add_argument("--loss_masks", required=True)
    parser.add_argument("--smooth", type=float, default=1., help="add k smoothing for length")
    args = parser.parse_args()

    length, cnt = load_pair(args.len_count, int, int)
    len_count = np.full(MAX_LEN + 1, args.smooth)
    for l, c in zip(length, cnt):
        len_count[l] += c
    len_count /= np.sum(len_count)
    len_nll = -np.log(len_count)

    loss, num_masks = load_pair(args.loss_masks, float, int)
    tot_loss = sum(len_nll[n] + n * l for l, n in zip(loss, num_masks))

    # comparable with per-token loss of Left2rightLM in predicting (w1, ..., wn, <eos>) from (<bos>, w1, ..., wn)
    num_tokens = len(num_masks) + sum(num_masks)
    avg_loss = tot_loss / num_tokens

    ppl = np.exp(avg_loss)
    print(f"ppl (overall) = {ppl:.2f}")


if __name__ == "__main__":
    main()