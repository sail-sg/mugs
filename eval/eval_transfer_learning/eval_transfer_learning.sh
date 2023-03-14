#!/bin/bash

##================= Transfer Learning ... ======

##================= CIFAR10 ... ==================
DATASET_ROOT=/dataset/CIFAR10/  

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 3.0e-6 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set CIFAR10 

## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.50e-06 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set CIFAR10 


##================= CIFAR100 ... ==================
DATASET_ROOT=/dataset/CIFAR100/ #/raid/common/imagenet-raw/

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.50e-06 \
    --weight-decay 0.02 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set CIFAR 

## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.50e-06 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set CIFAR 

##================= INA18 ... ==================
DATASET_ROOT=/dataset/2018/  

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 3.0e-5 \
    --weight-decay 0.05 \
    --epochs 360 \
    --reprob 0.1 \
    --data_set INAT 


## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 1.5e-5 \
    --weight-decay 0.05 \
    --epochs 360 \
    --reprob 0.1 \
    --data_set INAT 


##================= INA19 ... ==================
DATASET_ROOT=/dataset/2019/  

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.5e-5 \
    --weight-decay 0.05 \
    --epochs 360 \
    --reprob 0.1 \
    --data_set INAT19 


## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 3.0e-5 \
    --weight-decay 0.05 \
    --epochs 360 \
    --reprob 0.1 \
    --data_set INAT19 


##================= Flower ... ==================
DATASET_ROOT=/dataset/Flower/ 

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.50e-05 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set flowers 
 
## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 7.50e-05 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set flowers 


##================= Cars ... ==================
DATASET_ROOT=/dataset/Cars/  

## ViT-small
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_small \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 1.50e-04 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --data_set Cars 


## ViT-base
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --resume checkpoint.pth \
    --arch vit_base \
    --batch-size $BATCH_SIZE_PER_GPU \
    --lr 1.50e-04 \
    --weight-decay 0.05 \
    --epochs 1000 \
    --reprob 0.1 \
    --drop-path 0.3 \
    --data_set Cars 
