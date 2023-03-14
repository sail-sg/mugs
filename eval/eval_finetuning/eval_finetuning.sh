#!/bin/bash
DATASET_ROOT=/dataset/imageNet100_sicy/train/ #/raid/common/imagenet-raw/

## fine-tune pretrained ViT-small for 800 epochs
CHECKPOINT=./exps/vit_small_800ep.pth #pretrained model
OUTPUT=./exps/vit_small_800ep_weight.pth #backbone weights
# Step 1. extract the backbone weight 
python ./eval_finetuning/extract_backbone_weights_for_finetuning.py \
    --checkpoint $CHECKPOINT \
    --output $OUTPUT \
    --checkpoint_key teacher

# Step 2. load and fine-tune the backbone weight for 200 epochs
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=256
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_finetuning/eval_finetuning.py \
    --data_path $DATASET_ROOT \
    --finetune $OUTPUT \
    --model vit_small \
    --epochs 200 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_epochs 20 \
    --drop_path 0.1 \
    --lr 0.0012 \
    --layer_decay 0.55 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --layer_scale_init_value 0.0 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --use_cls \
    --imagenet_default_mean_and_std


## fine-tune pretrained ViT-base for 100 epochs
CHECKPOINT=./exps/vit_base_400ep.pth #pretrained model
OUTPUT=./exps/vit_base_400ep_weight.pth #backbone weights
# Step 1. extract the backbone weight 
python ./eval_finetuning/extract_backbone_weights_for_finetuning.py \
    --checkpoint $CHECKPOINT \
    --output $OUTPUT \
    --checkpoint_key teacher

# Step 2. load and fine-tune the backbone weight for 100 epochs
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=128
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_finetuning/eval_finetuning.py \
    --data_path $DATASET_ROOT \
    --finetune $OUTPUT \
    --model vit_base \
    --epochs 100 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_epochs 20 \
    --drop_path 0.2 \
    --lr 0.0012 \
    --layer_decay 0.55 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --layer_scale_init_value 0.0 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --use_cls \
    --imagenet_default_mean_and_std


## fine-tune pretrained ViT-large for 50 epochs
CHECKPOINT=./exps/vit_large_250ep.pth #pretrained model
OUTPUT=./exps/vit_large_250ep_weight.pth #backbone weights
# Step 1. extract the backbone weight 
python ./eval_finetuning/extract_backbone_weights_for_finetuning.py \
    --checkpoint $CHECKPOINT \
    --output $OUTPUT \
    --checkpoint_key teacher

# Step 2. load and fine-tune the backbone weight for 50 epochs
NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=64
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_finetuning/eval_finetuning.py \
    --data_path $DATASET_ROOT \
    --finetune $OUTPUT \
    --model vit_large \
    --epochs 50 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_epochs 5 \
    --drop_path 0.3 \
    --lr 0.0008 \
    --layer_decay 0.75 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --layer_scale_init_value 0.0 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --use_cls \
    --imagenet_default_mean_and_std