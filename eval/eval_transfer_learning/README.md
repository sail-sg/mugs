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


### Transfer learning on CIFAR10, CIFAR100, INAT18, INAT19, Flowers and Cars
To evaluate semi-supervised classification on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_transfer_learning.sh` 
or run one command in the `eval_transfer_learning.sh` which contains transfer learning on all models.
#### CIFAR10
For ViT-S/16, you can run 
```
DATASET_ROOT=/dataset/CIFAR10/  
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_small_800ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL --resume checkpoint.pth --arch vit_small --batch-size $BATCH_SIZE_PER_GPU --lr 3.0e-6 --weight-decay 0.05 --epochs 1000 --reprob 0.1 --data_set CIFAR10 
```
For ViT-B/16, you can run 
```
DATASET_ROOT=/dataset/CIFAR10/  
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=192
PRETRAINED_MODEL=./exps/vit_base_400ep_weight.pth
python -m torch.distributed.launch --nproc_per_node=4 ./eval_transfer_learning/eval_transfer_learning.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL --resume checkpoint.pth --arch vit_base --batch-size $BATCH_SIZE_PER_GPU --lr 7.50e-06 --weight-decay 0.05 --epochs 1000 --reprob 0.1 --data_set CIFAR10 
```
For other datasets, please find the command in the `eval_transfer_learning.sh`.

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