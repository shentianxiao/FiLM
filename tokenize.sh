#!/bin/bash

dataset=$1
tokenizer=$2

if [ $dataset == "wikitext-103" ]; then
    python -m data_process.text2id --dir data/$dataset --tokenizer $tokenizer --sentence_split
else
    python -m data_process.text2id --dir data/$dataset --tokenizer $tokenizer
fi

for split in "train" "valid"; do
    python -m data_process.len_count --file data/$dataset/$tokenizer/$split.id
done