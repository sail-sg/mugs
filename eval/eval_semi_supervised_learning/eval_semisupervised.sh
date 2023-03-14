#!/bin/bash
DATASET_ROOT=/dataset/imageNet100_sicy/train/

## semi-supervised learning for pretrained ViT-small on 1% labeled data via fine-tuning
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --class_num 1000 \
    --target_list_path ./eval_semi_supervised_learning/subset/1percent.txt


## semi-supervised learning for pretrained ViT-small on 10% labeled data via fine-tuning
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_semi_supervised_learning/eval_semi_supervised_learning.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --avgpool_patchtokens 0 \
    --finetune_head_layer 1 \
    --checkpoint_key teacher \
    --epochs 1000 \
    --opt adamw \
    --batch_size $BATCH_SIZE_PER_GPU \
    --lr 5.0e-6 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --class_num 1000 \
    --target_list_path ./eval/eval_semi_supervised_learning/subset/10percent.txt
   

## semi-supervised learning for pretrained ViT-small on 1% labeled data via logistic regression
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python ./eval_semi_supervised_learning/eval_logistic_regression.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --class_num 1000 \
    --lr_lambd 0.01 0.03 0.06 0.10 0.15 0.2 0.3 \
    --adjust_lambd 1 \
    --avgpool_patchtokens 0 \
    --multi_scale 0 \
    --target_list_path ./eval_semi_supervised_learning/subset/1percent.txt  

 
## semi-supervised learning for pretrained ViT-small on 10% labeled data via logistic regression
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python ./eval_semi_supervised_learning/eval_logistic_regression.py \
    --data_path $DATASET_ROOT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_small \
    --class_num 1000 \
    --lr_lambd 0.01 0.03 0.06 0.10 0.15 0.2 0.3 \
    --adjust_lambd 1 \
    --avgpool_patchtokens 0 \
    --multi_scale 0 \
    --target_list_path ./eval_semi_supervised_learning/subset/10percent.txt