from argparse import ArgumentParser

import evaluate

from utils.io import load_id


def load(path):
    data = load_id(path)
    return [" ".join(str(x) for x in ids) for ids in data]


def main():
    parser = ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--gen", required=True)
    args = parser.parse_args()

    gens = load(args.gen)
    refs = load(args.ref)

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")


if __name__ == "__main__":
    main()