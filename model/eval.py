import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import lightning as L

from utils.data import load_and_batch
from utils.operations import reorder
from . import get_model


def main():
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--input", default="")

    parser.add_argument("--order", default="left2right", choices=[
        "random", "left2right", "right2left", "min-entropy", "max-entropy"]) 

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1111)

    args = parser.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    model = get_model(args.checkpoint)
    args.task = "eval"
    model.hparams_test = args

    data, order = load_and_batch(args.target, model.bos_id, model.eos_id, model.pad_id, max_tokens=args.max_tokens)
    if args.input:
        input, _ = load_and_batch(args.input, model.bos_id, model.eos_id, model.pad_id, max_tokens=args.max_tokens)
        data = list(zip(input, data))
    dl = DataLoader(data, batch_size=1, num_workers=4)

    trainer = L.Trainer(accelerator="gpu", devices=args.gpus)
    trainer.test(model, dataloaders=dl)

    print(f"ppl (given length) = {model.nll.exp():.2f}")

    loss_masks = reorder(model.loss_masks, order)
    output_dir = f"{args.checkpoint}/eval"
    os.makedirs(output_dir, exist_ok=True)
    file = os.path.basename(args.input).replace("mask", "infill").replace(".id", f".{args.order}.loss_masks")
    with open(f"{output_dir}/{file}", "w") as f:
        for loss, num_masks in loss_masks:
            f.write(f"{loss:.6f}\t{int(num_masks)}\n")


if __name__ == "__main__":
    main()
