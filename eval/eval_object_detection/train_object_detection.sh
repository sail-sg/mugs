PRETRAINED=/path/to/pretrained/vit_small/checkpoint.pth
WEIGHT_FILE=/path/to/extracted/checkpoint_teacher.pth

if [ ! -f $WEIGHT_FILE ]; then
    python3 extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key teacher
fi

GPUS=8
CONFIG=vit_small_giou_4conv1f_coco_1x
OUTPUT_DIR=/path/to/det/output/dir
python3 -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=${PORT:-29500} \
    train.py \
    configs/cascade_rcnn/${CONFIG}.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
    model.pretrained=$WEIGHT_FILE \
    data.samples_per_gpu=4