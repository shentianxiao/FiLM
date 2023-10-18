import os
import json
import yaml


def load_str(path, split=True):
    data = []
    with open(path) as f:
        for line in f:
            item = line.split() if split else line.rstrip()
            data.append(item)
    return data


def load_id(path):
    data = []
    with open(path) as f:
        for line in f:
            item = line.split()
            data.append([int(x) for x in item])
    return data


def load_float(path):
    data = []
    with open(path) as f:
        for line in f:
            item = line.split()
            data.append([float(x) for x in item])
    return data


def load_pair(path, type1, type2):
    x, y = [], []
    with open(path) as f:
        for line in f:
            parts = line.split("\t")
            x.append(type1(parts[0]))
            y.append(type2(parts[1]))
    return x, y


def load_text(path):
    with open(path) as f:
        data = json.load(f)
    return [item["text"] for item in data]


def write_str(data, path, join=True):
    with open(path, "w") as f:
        for item in data:
            line = " ".join(item) if join else item
            f.write(line + "\n")


def write_id(data, path):
    with open(path, "w") as f:
        for item in data:
            line = " ".join(str(x) for x in item)
            f.write(line + "\n")


def write_float(data, path):
    with open(path, "w") as f:
        for item in data:
            line = " ".join(f"{x:.3f}" for x in item)
            f.write(line + "\n")


def write_text(data, path):
    data = [{"index": id + 1, "text": item} for id, item in enumerate(data)]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_hparams(path):
    with open(f"{path}/hparams.yaml") as f:
        return yaml.safe_load(f)


def get_checkpoint(path):
    ckpts = os.listdir(f"{path}/checkpoints")
    ckpt = sorted(ckpts)[0]
    print(f"Load from {ckpt}")
    return f"{path}/checkpoints/{ckpt}"