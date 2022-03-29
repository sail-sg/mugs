#!/bin/bash
DATASET_ROOT=/dataset/imageNet100_sicy/train/ #/raid/common/imagenet-raw/

## train ViT-small for 100 epochs
OUTPUT_ROOT=./exps/vit_small_100ep
NPROC_PER_NODE=8 # GPU numbers
BATCH_SIZE_PER_GPU=64
DEBUG=false # debug = true, then we only load subset of the whole training dataset
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE main.py \
	--data_path $DATASET_ROOT \
	--output_dir $OUTPUT_ROOT \
	--arch vit_small \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.04 \
	--group_warmup_teacher_temp_epochs 0 \
	--norm_last_layer false \
	--norm_before_pred true \
	--batch_size_per_gpu $BATCH_SIZE_PER_GPU \
	--epochs 100 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0008 \
	--min_lr 1e-06 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.1 \
	--weight_decay 0.04 \
	--weight_decay_end 0.2 \
	--freeze_last_layer 1 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug $DEBUG

## train ViT-small for 300 epochs
OUTPUT_ROOT=./exps/vit_small_300ep
NPROC_PER_NODE=16 # GPU numbers
BATCH_SIZE_PER_GPU=64
DEBUG=false # debug = true, then we only load subset of the whole training dataset
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE main.py \
	--data_path $DATASET_ROOT \
	--output_dir $OUTPUT_ROOT \
	--arch vit_small \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.07 \
	--group_warmup_teacher_temp_epochs 30 \
	--norm_last_layer false \
	--norm_before_pred true \
	--batch_size_per_gpu $BATCH_SIZE_PER_GPU \
	--epochs 300 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0008 \
	--min_lr 1e-06 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.1 \
	--weight_decay 0.04 \
	--weight_decay_end 0.1 \
	--freeze_last_layer 1 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug $DEBUG

## train ViT-small for 800 epochs
NPROC_PER_NODE=16 # GPU numbers
BATCH_SIZE_PER_GPU=64
DEBUG=false # debug = true, then we only load subset of the whole training dataset
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE main.py \
	--data_path $DATASET_ROOT \
	--output_dir $OUTPUT_ROOT \
	--arch vit_small \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.07 \
	--group_warmup_teacher_temp_epochs 30 \
	--norm_last_layer false \
	--norm_before_pred true \
	--batch_size_per_gpu $BATCH_SIZE_PER_GPU \
	--epochs 800 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0008 \
	--min_lr 1e-06 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.1 \
	--weight_decay 0.04 \
	--weight_decay_end 0.1 \
	--freeze_last_layer 1 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug $DEBUG

## train ViT-base for 400 epochs
OUTPUT_ROOT=./exps/vit_base_400ep
NPROC_PER_NODE=24 # GPU numbers
BATCH_SIZE_PER_GPU=42
DEBUG=false # debug = true, then we only load subset of the whole training dataset
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE main.py \
	--data_path $DATASET_ROOT \
	--output_dir $OUTPUT_ROOT \
	--arch vit_base \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.07 \
	--group_warmup_teacher_temp_epochs 50 \
	--norm_last_layer false \
	--norm_before_pred true \
	--batch_size_per_gpu $BATCH_SIZE_PER_GPU \
	--epochs 400 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0008 \
	--min_lr 2e-06 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.1 \
	--weight_decay 0.04 \
	--weight_decay_end 0.1 \
	--freeze_last_layer 3 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug $DEBUG

## train ViT-large for 250 epochs
OUTPUT_ROOT=./exps/vit_large_250ep
NPROC_PER_NODE=40 # GPU numbers
BATCH_SIZE_PER_GPU=16
DEBUG=false # debug = true, then we only load subset of the whole training dataset
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE main.py \
	--data_path $DATASET_ROOT \
	--output_dir $OUTPUT_ROOT \
	--arch vit_large \
	--instance_queue_size 65536 \
	--local_group_queue_size 65536 \
	--use_bn_in_head false \
	--instance_out_dim 256 \
	--instance_temp 0.2 \
	--local_group_out_dim 256 \
	--local_group_temp 0.2 \
	--local_group_knn_top_n 8 \
	--group_out_dim 65536 \
	--group_student_temp 0.1 \
	--group_warmup_teacher_temp 0.04 \
	--group_teacher_temp 0.07 \
	--group_warmup_teacher_temp_epochs 50 \
	--norm_last_layer true \
	--norm_before_pred true \
	--batch_size_per_gpu $BATCH_SIZE_PER_GPU \
	--epochs 250 \
	--warmup_epochs 10 \
	--clip_grad 3.0 \
	--lr 0.0015 \
	--min_lr 1.5e-4 \
	--patch_embed_lr_mult 0.2 \
	--drop_path_rate 0.3 \
	--weight_decay 0.025 \
	--weight_decay_end 0.08 \
	--freeze_last_layer 3 \
	--momentum_teacher 0.996 \
	--use_fp16 false \
	--local_crops_number 10 \
	--size_crops 96 \
	--global_crops_scale 0.25 1 \
	--local_crops_scale 0.05 0.25 \
	--timm_auto_augment_par rand-m9-mstd0.5-inc1 \
	--prob 0.5 \
	--use_prefetcher true \
	--debug $DEBUG
