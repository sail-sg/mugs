#!/bin/bash

## Video segmentation by using ViT-base
DATASET_ROOT=/dataset/video/  
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
OUTPUT=./exps/vit_small_800ep_video/

python ./eval_video/eval_video.py \
    --data_path $DATASET_ROOT \
    --output_dir $OUTPUT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_base \
    --topk 5 \
    --n_last_frames 20 \
    --size_mask_neighborhood 12
