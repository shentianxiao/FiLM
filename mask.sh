#!/bin/bash

dataset=$1
tokenizer=$2
split=("valid" "test")

python -m data_process.mask_all --dir data/$dataset/$tokenizer

if [ $dataset == "roc_stories" ]; then
    python -m data_process.mask_roc --tokenizer $tokenizer
    mask=("span1")
else
    python -m data_process.mask_rand --dir data/$dataset/$tokenizer
    mask=("span1-5" "span1" "span2" "span3" "span4" "span5")
fi

for s in "${split[@]}"; do
    for m in "${mask[@]}"; do
        python -m data_process.extract --mask data/$dataset/$tokenizer/mask/$s.mask.$m.id --infill data/$dataset/$tokenizer/mask/$s.infill.$m.id
        python -m data_process.id2text --tokenizer $tokenizer --file data/$dataset/$tokenizer/mask/$s.mask.$m.id
        python -m data_process.id2text --tokenizer $tokenizer --file data/$dataset/$tokenizer/mask/$s.fill.$m.id
    done
done