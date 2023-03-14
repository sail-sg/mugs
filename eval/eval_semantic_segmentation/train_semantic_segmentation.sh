PRETRAINED=/path/to/pretrained/vit_small/checkpoint.pth
WEIGHT_FILE=/path/to/extracted/checkpoint_teacher.pth

if [ ! -f $WEIGHT_FILE ]; then
    python3 extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key teacher
fi


GPUS=4
CONFIG=vit_small_512_ade20k_160k
OUTPUT_DIR=/path/to/seg/output/dir

python3 -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=${PORT:-29500} \
    train.py \
    configs/upernet/${CONFIG}.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --deterministic \
    --options model.pretrained=$WEIGHT_FILE \
    data.samples_per_gpu=4 \
    model.backbone.out_with_norm=true