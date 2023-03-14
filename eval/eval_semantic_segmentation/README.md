# Mugs: A Multi-Granular Self-Supervised Learning Framework
Here we provide the evaluation code to evaluate the pretrained model by **Mugs** on several downstream tasks.

### Environment
For reproducing, please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset.
This codebase has been developed with python version 3.8, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. For the full 
environment, please refer to our `Dockerfile` file. 


## Evaluation
For all downstream tasks tested in the manuscript, you can directly use the corresponding `.sh` evaluation file 
in the `eval` fold. For these `.sh` evaluation file, all hyper-parameters are assigned. In this way, what all you 
is to assign the paths of the dataset and the pretrained model. 


### Segmantic Segmentation
#### Environment
For object detection and semantic segmentation, we mainly use `apex`, `mmcv-full==1.2.7`, [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and `mmsegmentation==0.11.0`.

For PyTorch 1.7.1, CUDA 11.0 and CUDNN 8, you can use this [Dockerfile](Dockerfile) or the following commands to set up the environment.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

pip install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.2.7
pip install --no-cache-dir pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops


git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip install --no-cache-dir -r requirements/build.txt
pip install --no-cache-dir -v -e .
cd ..


git clone -b v0.11.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip install --no-cache-dir -v -e .
cd ..
```

#### Evaluation
Enter this dir and run (as shown in [eval_semantic_segmentation.sh](eval_semantic_segmentation.sh)):
```bash
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
```

#### Training
Enter this dir and run (as shown in [train_semantic_segmentation.sh](train_semantic_segmentation.sh)):
```bash
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
```


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@inproceedings{mugs2022SSL,
  title={Mugs: A Multi-Granular Self-Supervised Learning Framework},
  author={Pan Zhou and Yichen Zhou and Chenyang Si and Weihao Yu and Teck Khim Ng and Shuicheng Yan},
  booktitle={Axriv},
  year={2022}
}
```