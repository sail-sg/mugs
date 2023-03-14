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

### Linear probing classification on ImageNet-1K
To evaluate linear probing on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_linear.sh` 
or run one command in the `eval_linear.sh` which contains linear probing evaluations on all models:
```
python -m torch.distributed.launch --nproc_per_node=4 ./eval_linear_probing/eval_linear.py --data_path $DATASET_ROOT  --pretrained_weights 
$PRETRAINED_WEIGHTS  --checkpoint_key teacher  --arch vit_small  --epochs 100  --n_last_blocks 4  --avgpool_patchtokens False  --batch_size_per_gpu 256  --lr 0.015  --lr_min 0.00001
```
Note for ViT-Large model, we follow iBOT and build many classifiers to evaluate its linear probing performance. 
You can run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 ./eval_linear_probing/eval_linear_multi_classifier.py --data_path $DATASET_ROOT 
--pretrained_weights $PRETRAINED_WEIGHTS --checkpoint_key teacher --arch vit_large --epochs 100 --n_last_blocks 1 --avgpool_patchtokens 2 
--color_aug false --sweep_lr_only False --batch_size_per_gpu 128 --lr 0.015 --lr_min 0.00001 
```
For all linear probing logs, please refer to the `eval logs` column of [Table 1](https://github.com/sail-sg/mugs) or the following `Table 2`. 


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