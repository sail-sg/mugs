# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
implement different datasets. 
"""
import numpy as np
import torch
import torch.distributed as dist
import json
import os
import math
import scipy.io as scio


from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import ImageFolder, default_loader


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

class CarsDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        # pdb.set_trace()
        data = scio.loadmat(os.path.join(root, \
            f'cars_annos.mat'))['annotations'][0].tolist()
        data = [elem for elem in data if elem[-1].item() == int(not train)]

        targeter = {}
        indexer = 0
        for elem in data:
            catg = elem[4].item()
            if catg not in targeter.keys():
                targeter[catg] = indexer
                indexer += 1
        self.nb_classes = len(targeter)


        self.samples = []
        for elem in data:
            catg, path = elem[5].item(), elem[0].item()
            path = os.path.join(root, path)
            self.samples.append((path, catg))


class FlwrsDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        # pdb.set_trace()
        data = np.array(sorted(os.listdir(os.path.join(root, 'jpg'))))
        labels = scio.loadmat(os.path.join(root, f'imagelabels.mat'))['labels'][0]
        ids = scio.loadmat(os.path.join(root, f'setid.mat'))        
        ids = np.concatenate((ids['trnid'], ids['valid']), axis=1)[0] if train else ids['tstid'][0]
        
        labels -= 1
        ids -= 1

        targeter = {}
        indexer = 0
        for catg in labels[ids]:
            if catg not in targeter.keys():
                targeter[catg] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for path, catg in zip(data[ids], labels[ids]):
            path = os.path.join(root, 'jpg', path)
            self.samples.append((path, catg))

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    # pdb.set_trace()
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'Cars':
        dataset = CarsDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = dataset.nb_classes
    else:
        dataset = FlwrsDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = dataset.nb_classes
    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch