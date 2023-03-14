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

### K-NN classification on ImageNet-1K
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_knn.sh` 
or run one command in the `eval_knn.sh` which contains KNN evaluations on all models:
```
python -m torch.distributed.launch --nproc_per_node=1 ./eval_knn/eval_knn.py --arch vit_small --temperature $TEMPERATURE --batch_size_per_gpu $BATCH_SIZE_PER_GPU --pretrained_weights $PRETRAINED_WEIGHTS --data_path $DATASET_ROOT 
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