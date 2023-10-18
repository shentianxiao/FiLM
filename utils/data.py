import numpy as np

from transformers import AutoTokenizer

from .io import load_id


MAX_LEN = 510  # account for extra special tokens


def get_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if "gpt2" in name:
        tokenizer.add_special_tokens({"mask_token": "<mask>", "pad_token": "<pad>"})
    return tokenizer


def tokenize(tokenizer, text):
    ids = tokenizer(text)["input_ids"]
    if ids[0] == tokenizer.bos_token_id:
        ids.pop(0)
    if ids[-1] == tokenizer.eos_token_id:
        ids.pop()
    return ids


def add_bos_eos(data, bos_id, eos_id):
    for i in range(len(data)):
        if bos_id:
            data[i].insert(0, bos_id)
        if eos_id:
            data[i].append(eos_id)


def remove_special_tokens(data, bos_id, eos_id, pad_id):
    for i in range(len(data)):
        while data[i] and data[i][-1] == pad_id:
            data[i].pop()
        if data[i] and data[i][0] == bos_id:
            data[i].pop(0)
        if data[i] and data[i][-1] == eos_id:
            data[i].pop()


def get_batches(data, pad_id, max_tokens=None, batch_size=None, same_len=False):
    assert (max_tokens is None) ^ (batch_size is None)

    # sort data by length so that examples with similar lengths are batched together to reduce pads
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda x: len(x[1]), reverse=True)
    order, data = zip(*z)

    # construct batches
    batches = []
    i = 0
    while i < len(data):
        j = i
        l = len(data[i])
        while j < len(data) and (len(data[j]) == l or not same_len) and \
            ((max_tokens and l * (j-i+1) <= max_tokens) or (batch_size and j-i+1 <= batch_size)):
            j += 1
        batch = np.stack([x + [pad_id] * (l - len(x)) for x in data[i: j]])
        batches.append(batch)
        i = j
    return batches, order


def load_and_batch(path, bos_id, eos_id, pad_id, max_tokens=None, batch_size=None, same_len=False):
    data = load_id(path)
    add_bos_eos(data, bos_id, eos_id)
    return get_batches(data, pad_id, max_tokens, batch_size, same_len)