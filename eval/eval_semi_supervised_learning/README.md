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


### Semi-supervised classification on ImageNet-1K
To evaluate semi-supervised classification on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_semisupervised.sh` 
or run one command in the `eval_semisupervised.sh` which contains semi-supervised classifications on all models.
#### fine-tuning
For 1% labeled data, you can run 
```
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./pretrained_model_path
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_semi_supervised_learning/eval_semi_supervised_learning.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL --arch vit_small --avgpool_patchtokens 0 --finetune_head_layer 1 --checkpoint_key teacher --epochs 1000 --opt adamw --batch_size $BATCH_SIZE_PER_GPU --lr 5.0e-6 --weight-decay 0.05 --drop-path 0.1 --class_num 1000 --target_list_path ./eval_semi_supervised_learning/subset/1percent.txt
```
For 10% labeled data, you can run 
```
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./pretrained_model_path
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_semi_supervised_learning/eval_semi_supervised_learning.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL -arch vit_small --avgpool_patchtokens 0 --finetune_head_layer 1 --checkpoint_key teacher --epochs 1000 --opt adamw --batch_size $BATCH_SIZE_PER_GPU --lr 5.0e-6 --weight-decay 0.05 --drop-path 0.1 --class_num 1000 --target_list_path ./eval/eval_semi_supervised_learning/subset/10percent.txt
```
#### logistic regression
For 1% labeled data, you can run 
```
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./pretrained_model_path
python ./eval_semi_supervised_learning/eval_logistic_regression.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL --arch vit_small --class_num 1000 --lr_lambd 0.01 0.03 0.06 0.10 0.15 --adjust_lambd 1 --avgpool_patchtokens 0 --multi_scale 0 --target_list_path ./eval_semi_supervised_learning/subset/1percent.txt  
```
For 10% labeled data, you can run 
```
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=128
PRETRAINED_MODEL=./pretrained_model_path
python ./eval_semi_supervised_learning/eval_logistic_regression.py --data_path $DATASET_ROOT --pretrained_weights $PRETRAINED_MODEL --arch vit_small --class_num 1000 -lr_lambd 0.01 0.03 0.06 0.10 0.15 --adjust_lambd 1 --avgpool_patchtokens 0 --multi_scale 0 --target_list_path ./eval_semi_supervised_learning/subset/10percent.txt
```
For all semi-supervised learning logs on ViT-S/16, please find them in the above `Table 2`. 

**<p align="center">Table 2. Hyper-parameters, logs and model weights for linear probing, fine-tuning.</p>** 
<table>
  <tr>
    <th>arch</th>
    <th colspan="3">fine-tuning</th>
    <th colspan="3">logistic regression</th>
  </tr>
  <tr>
    <td>1%</td>
    <td>66.8%</td>
    <td><a href="https://drive.google.com/file/d/1WvzecnCwbcRGfXNUdPKYGIyExQ45J2gw/view?usp=sharing">fine-tune weights</a></td>
    <td><a href="https://drive.google.com/file/d/1-9LyX3BP6Xz0ErVI3eu_efe1TjwauSxZ/view?usp=sharing">fine-tune logs</a></td>
    <td>66.9%</td>
    <td><a href="https://drive.google.com/file/d/1GgFjxSjWn1w_CRuEbP3YZxezUL0McguR/view?usp=sharing">logistic weights</a></td>
    <td><a href="https://drive.google.com/file/d/1XaszkEPwHnVc1UgIs7EkGINpuD7GKoOl/view?usp=sharing">logistic logs</a></td>
  </tr>
  <tr>
    <td>10%</td>
    <td>76.8%</td>
    <td><a href="https://drive.google.com/file/d/1URX5u1jGcxSmurfCwokDQw6Psmdq2FuR/view?usp=sharing">fine-tune weights</a></td>
    <td><a href="https://drive.google.com/file/d/1k4cmTreajsIlxlF6-CURQ9e2pZc9Ka6-/view?usp=sharing">fine-tune logs</a></td>
    <td>74.0%</td>
    <td><a href="https://drive.google.com/file/d/1dxHsoTea8tHCEfbt_1Zw4cc61g16QsPZ/view?usp=sharing">logistic weights</a></td>
    <td><a href="https://drive.google.com/file/d/1Q0AheJ9NLrko4VurhpZnZEv-4SEakaOz/view?usp=sharing">logistic logs</a></td>
  </tr>
</table>

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