#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=$HOME/Megatron-LM/data/my-bert_text_sentence
CHECKPOINT_PATH=$HOME/Megatron-LM/ckpt/bert-$(date +%y%m%d-%H%M%S)

python pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 256 \
       --num-attention-heads 4 \
       --batch-size 1 \
       --seq-length 128 \
       --max-position-embeddings 512 \
       --train-iters 2000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $HOME/Megatron-LM/data/bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10
