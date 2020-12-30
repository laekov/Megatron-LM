#!/bin/bash

python tools/preprocess_data.py \
	--input $HOME/datas/wikip.json \
	--output-prefix my-bert \
	--vocab $HOME/datas/bert-large-uncased-vocab.txt \
	--dataset-impl mmap \
	--tokenizer-type BertWordPieceLowerCase \
	--split-sentences
