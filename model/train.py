from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.data import load_and_batch
from . import FiLM


def main():
    parser = ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--save_dir", default="checkpoints/")

    parser.add_argument("--pretrained_model", default="roberta-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--accum_grad", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=100000)

    parser.add_argument("--weight_func", default="beta", choices=["beta", "triangular", "delta"])
    parser.add_argument("--weight_param", type=float, nargs="+", default=[1, 1])
    parser.add_argument("--valid_samples", type=int, default=10, help=
                        "sample multiple times to reduce variance in loss calculation")

    parser.add_argument("--precision", default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1111)

    args = parser.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    model = FiLM(args)

    train_data, _ = load_and_batch(args.train, model.bos_id, model.eos_id, model.pad_id, max_tokens=args.max_tokens)
    valid_data, _ = load_and_batch(args.valid, model.bos_id, model.eos_id, model.pad_id, max_tokens=args.max_tokens)

    # since we have batched data by max_tokens, batch_size is set to 1 in DataLoader 
    # access the original batch with batch[0]
    train_dl = DataLoader(train_data, batch_size=1, num_workers=4, persistent_workers=True, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=1, num_workers=4, persistent_workers=True)

    trainer = L.Trainer(
        logger=TensorBoardLogger(args.save_dir),
        callbacks=[ModelCheckpoint(monitor="valid_loss", save_last=True)],
        accumulate_grad_batches=args.accum_grad,
        max_steps=args.train_steps,
        precision=args.precision,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu", 
        devices=args.gpus
        )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


if __name__ == "__main__":
    main()
