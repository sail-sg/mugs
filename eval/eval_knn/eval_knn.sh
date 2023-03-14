#!/bin/bash
DATASET_ROOT=/dataset/imageNet100_sicy/ #/raid/common/imagenet-raw/


## fine-tune ViT-small for 800 epochs
PRETRAINED_WEIGHTS=./exps/vit_small_800ep/checkpoint.pth
NPROC_PER_NODE=1
BATCH_SIZE_PER_GPU=256

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_knn/eval_knn.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --arch vit_small \
    --temperature 0.05 \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU

## fine-tune ViT-base for 400 epochs
PRETRAINED_WEIGHTS=./exps/vit_base_400ep/checkpoint.pth
NPROC_PER_NODE=1
BATCH_SIZE_PER_GPU=256

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_knn/eval_knn.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --arch vit_base \
    --temperature 0.04 \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU
 

## fine-tune ViT-large for 250 epochs
PRETRAINED_WEIGHTS=./exps/vit_large_250ep/checkpoint.pth
NPROC_PER_NODE=1
BATCH_SIZE_PER_GPU=128

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_knn/eval_knn.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --arch vit_large \
    --temperature 0.04 \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU