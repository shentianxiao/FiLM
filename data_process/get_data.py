import os
import requests
import csv
from argparse import ArgumentParser

from datasets import load_dataset

from utils.io import write_text


def load_wiki103():
    data = load_dataset("wikitext", "wikitext-103-raw-v1")
    for split in data.keys():
        data[split] = data[split]["text"]
    data["valid"] = data.pop("validation")

    for split in data.keys():
        texts = [text for text in data[split] if text]
        data[split] = []
        for text in texts:
            if text.startswith(" = ") and not text.startswith(" = = "):
                data[split].append("")
            data[split][-1] += text
    return data


def load_lm1b(n_valid=300000):
    data = load_dataset("lm1b")
    for split in data.keys():
        data[split] = data[split]["text"]
    data["train"], data["valid"] = data["train"][:-n_valid], data["train"][-n_valid:]
    return data


def download_rocstories():
    dir = "data/roc_stories/raw"
    if not os.path.exists(dir):
        os.makedirs(dir)
        url = {"winter2017": "https://goo.gl/0OYkPK",
               "spring2016": "https://goo.gl/7R59b1",
               "spring2016_valid": "https://docs.google.com/spreadsheets/d/1FkdPMd7ZEw_Z38AsFSTzgXeiJoLdLyXY_0B_0JIJIbw/export?format=csv",
               "spring2016_test": "https://docs.google.com/spreadsheets/d/11tfmMQeifqP-Elh74gi2NELp0rx9JMMjnQ_oyGKqCEg/export?format=csv"}
        for key, val in url.items():
            response = requests.get(val)
            with open(f"{dir}/{key}.csv", "w") as f:
                f.write(response.text)


def load_rocstories_sentence():
    dir = "data/roc_stories/raw"
    data = {"train": [], "valid": [], "test": []}
    for key in ["winter2017", "spring2016"]:
        with open(f"{dir}/{key}.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                sents = row[2:]
                sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
                data["train"].append(sents)
    for key in ["valid", "test"]:
        with open(f"{dir}/spring2016_{key}.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
                sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
                data[key].append(sents)
    return data


def load_rocstories():
    download_rocstories()
    data = load_rocstories_sentence()
    for key in data.keys():
        data[key] = ["".join(sents) for sents in data[key]]
    return data


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["wikitext-103", "lm1b", "roc_stories"])
    args = parser.parse_args()

    if args.dataset == "wikitext-103":
        data = load_wiki103()
    elif args.dataset == "lm1b":
        data = load_lm1b()
    elif args.dataset == "roc_stories":
        data = load_rocstories()
    else:
        raise ValueError

    output_dir = f"data/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        write_text(data[split], f"{output_dir}/{split}.json")


if __name__ == "__main__":
    main()
