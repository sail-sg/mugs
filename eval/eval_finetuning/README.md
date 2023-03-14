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


### Fine-tuning classification on ImageNet-1K
To evaluate fine-tuning on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_finetuning.sh` 
or run one command in the `eval_finetuning.sh` which contains fine-tuning evaluations on all models.
#### Step 1. extract the backbone weight 
```
python ./eval_finetuning/extract_backbone_weights_for_finetuning.py --checkpoint $CHECKPOINT --output $OUTPUT --checkpoint_key teacher
```
#### Step 2. load and fine-tune the backbone weight 
```
NPROC_PER_NODE=4
BATCH_SIZE_PER_GPU=256
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE ./eval_finetuning/eval_finetuning.py --data_path $DATASET_ROOT --finetune $OUTPUT --model vit_small --epochs 200 --batch_size $BATCH_SIZE_PER_GPU --warmup_epochs 20 --drop_path 0.1 --lr 0.0012 --layer_decay 0.55 --mixup 0.8 --cutmix 1.0 --layer_scale_init_value 0.0 --disable_rel_pos_bias --abs_pos_emb --use_cls --imagenet_default_mean_and_std
```
For all fine-tuning logs, please find them in `Table 2`. 

**<p align="center">Table 2. Hyper-parameters, logs and model weights for linear probing, fine-tuning.</p>** 
<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>pretraining epochs</th>
    <th>k-nn</th>
    <th>linear</th>
    <th>fine-tune</th>
    <th colspan="2">linear evaluation</th>
    <th colspan="2">fine-tuning evaluation</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>800</td>
    <td>75.6%</td>
    <td>78.9%</td>
    <td>82.6%</td>
    <td><a href="https://drive.google.com/file/d/14LF-T94dCBqLii0qhOfZhZzZm6AuhCi_/view?usp=sharing">linear weights</a></td>
    <td><a href="https://drive.google.com/file/d/12tiO4glWZNB044TYiPPCfbnUX_9AbqVc/view?usp=sharing">eval logs</a></td>
    <td><a href="https://drive.google.com/file/d/1cEkQW72VZv-4aQVbQHP4CBgyPJP22CPv/view?usp=sharing">fine-tune weights</a></td>
    <td><a href="https://drive.google.com/file/d/1LrElU1T4lvHxCuU5LJ-llX-9cCMl8o1L/view?usp=sharing">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>400</td>
    <td>78.0%</td>
    <td>80.6%</td>
    <td>84.3%</td>
    <td><a href="https://drive.google.com/file/d/1MAz28bBgzPb7MVhfbveL7PTox06xu_Wx/view?usp=sharing">linear weights</a></td>
    <td><a href="https://drive.google.com/file/d/1gOR250QFLZfe40pLNPcOqaLPAnKLuE_C/view?usp=sharing">eval logs</a></td>
    <td><a href="https://drive.google.com/file/d/1YTC9rj5t8onqJ5oAmVPAcXa1tADQtaYe/view?usp=sharing">fine-tune weights</a></td>
    <td><a href="https://drive.google.com/file/d/1L8EixjzZzP62dU3Z6mzykIdjuVHU-bpb/view?usp=sharing">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>307M</td>
    <td>250</td>
    <td>80.3%</td>
    <td>82.1%</td>
    <td>85.2%</td>
    <td><a href="https://drive.google.com/file/d/1j6rQwFTsT3NMLBs4s6qrQbjxn-HK1Mv6/view?usp=sharing">linear weights</a></td>
    <td><a href="https://drive.google.com/file/d/1rqWenRFN0czat_55GY9GNOu7gS6fww3g/view?usp=sharing">eval logs</a></td>
    <td><a href="https://drive.google.com/file/d/10Tcp-EMkNz1Kj1enjTYoGG90jkH9Gx-7/view?usp=sharing">fine-tune weights</a></td>
    <td><a href="https://drive.google.com/file/d/16o19XGdwR9_lsGdJqMTBZACgHOONppx2/view?usp=sharing">eval logs</a></td>
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