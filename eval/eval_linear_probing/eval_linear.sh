#!/bin/bash
DATASET_ROOT=/dataset/imageNet100_sicy/ #/raid/common/imagenet-raw/

## fine-tune ViT-small for 100 epochs
PRETRAINED_WEIGHTS=./exps/vit_small_100ep/checkpoint.pth
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=256

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_linear_probing/eval_linear.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --checkpoint_key teacher \
    --arch vit_small \
    --epochs 100 \
    --n_last_blocks 4 \
    --avgpool_patchtokens False \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.015 \
    --lr_min 0.00001

 
## fine-tune ViT-small for 300 epochs
PRETRAINED_WEIGHTS=./exps/vit_small_300ep/checkpoint.pth
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=256

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_linear_probing/eval_linear.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --checkpoint_key teacher \
    --arch vit_small \
    --epochs 100 \
    --n_last_blocks 4 \
    --avgpool_patchtokens False \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.02 \
    --lr_min 0.0

## fine-tune ViT-small for 800 epochs
PRETRAINED_WEIGHTS=./exps/vit_small_800ep/checkpoint.pth
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=256

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_linear_probing/eval_linear.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --checkpoint_key teacher \
    --arch vit_small \
    --epochs 100 \
    --n_last_blocks 4 \
    --avgpool_patchtokens False \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.01 \
    --lr_min 0.0 

## fine-tune ViT-base for 400 epochs
PRETRAINED_WEIGHTS=./exps/vit_base_400ep/checkpoint.pth
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_linear_probing/eval_linear.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --checkpoint_key teacher \
    --arch vit_small \
    --epochs 100 \
    --n_last_blocks 1 \
    --avgpool_patchtokens True \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.002 \
    --lr_min 0.00001 


## fine-tune ViT-large for 250 epochs
PRETRAINED_WEIGHTS=./exps/vit_large_250ep/checkpoint.pth
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_linear_probing/eval_linear_multi_classifier.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --checkpoint_key teacher \
    --arch vit_large \
    --epochs 100 \
    --n_last_blocks 1 \
    --avgpool_patchtokens 2 \
    --color_aug false \
    --sweep_lr_only False \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --lr 0.015 \
    --lr_min 0.00001 
 