#!/bin/bash

source /opt/spack/share/spack/setup-env.sh
spack load python

export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

MP_SIZE=4

if [ -z $RANK ]
then
	RANK=0
	WORLD_SIZE=1
fi
DATA_PATH=$HOME/Megatron-LM/data/my-bert_text_sentence
CHECKPOINT_PATH=$HOME/Megatron-LM/ckpt/bert-$(date +%y%m%d-%H%M%S)

python pretrain_bert.py \
	   --model-parallel-size $MP_SIZE \
       --num-layers 6 \
       --hidden-size 512 \
       --num-attention-heads 8 \
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
