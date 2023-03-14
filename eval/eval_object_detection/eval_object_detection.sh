GPUS=8
CONFIG=vit_small_giou_4conv1f_coco_1x
DET_WEIGHT=/path/to/det/vit_small_giou_4conv1f_coco_1x.pth

python3 -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=${PORT:-29500} \
    test.py \
    configs/cascade_rcnn/${CONFIG}.py \
    $DET_WEIGHT \
    --launcher pytorch \
    --eval bbox segm \
    --cfg-options model.backbone.use_checkpoint=True \
    data.samples_per_gpu=4