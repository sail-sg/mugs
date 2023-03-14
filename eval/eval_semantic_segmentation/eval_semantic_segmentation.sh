GPUS=4
CONFIG=vit_small_512_ade20k_160k
SEG_WEIGHT=/path/to/vit_small_512_ade20k_160k.pth

python3 -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=${PORT:-29500} \
    test.py \
    configs/upernet/${CONFIG}.py \
    $SEG_WEIGHT \
    --launcher pytorch \
    --eval mIoU \
    --options \
    data.samples_per_gpu=4 \
    model.backbone.out_with_norm=true