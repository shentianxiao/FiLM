import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import lightning as L

from utils.data import load_and_batch, remove_special_tokens
from utils.io import write_id
from utils.operations import reorder
from . import get_model


def main():
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)

    parser.add_argument("--temp", type=float, default=1.)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--order", default="left2right", choices=[
        "random", "left2right", "right2left", "min-entropy", "max-entropy"])

    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1111)

    args = parser.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    model = get_model(args.checkpoint)
    args.task = "generate"
    model.hparams_test = args

    data, order = load_and_batch(args.input, model.bos_id, model.eos_id, model.pad_id, args.max_tokens, batch_size=None, same_len=False)
    dl = DataLoader(data, batch_size=1, num_workers=4)

    trainer = L.Trainer(accelerator="gpu", devices=args.gpus)
    trainer.test(model, dataloaders=dl)

    output_ids = reorder(model.outputs, order)
    remove_special_tokens(output_ids, model.bos_id, model.eos_id, model.pad_id)

    output_dir = f"{args.checkpoint}/generate"
    os.makedirs(output_dir, exist_ok=True)
    file = os.path.basename(args.input)
    file = file.replace("mask", "infill")
    write_id(output_ids, f"{output_dir}/{file}")


if __name__ == "__main__":
    main()
