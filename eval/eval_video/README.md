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


### Video object segmentation on DAVIS 2017 
To evaluate video object segmentation on a pre-trained model, you can first enter the `eval` fold. Then you can run `eval_video.sh` 
or run one command in the `eval_video.sh` which contains  video object segmentation on all models. Same as DINO, please verify that you're using pytorch version 1.7.1 since we are not able to reproduce the results with most recent pytorch 1.8.1 at 
the moment.

**Step 1: Prepare DAVIS 2017 data**  
```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
```

**Step 2: Video object segmentation**  
```
python ./eval_video/eval_video.py \
    --data_path $DATASET_ROOT \
    --output_dir $OUTPUT \
    --pretrained_weights $PRETRAINED_MODEL \
    --arch vit_base \
    --topk 5 \
    --n_last_frames 20 \
    --size_mask_neighborhood 12
```

**Step 3: Evaluate the obtained segmentation**  
```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python $HOME/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /path/to/saving_dir --davis_path $HOME/davis-2017/DAVIS/
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