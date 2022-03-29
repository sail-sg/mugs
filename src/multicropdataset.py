# Copyright 2022 Garena Online Private Limited
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
multi-crop dataset to implement multi-crop augmentation and also dataset
"""
import copy
import random

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from src.dataset import ImageFolder
from src.RandAugment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import _pil_interp


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def strong_transforms(
    img_size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="random",
    use_prefetcher=True,
    mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
    std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    re_num_splits=0,
    color_aug=False,
    strong_ratio=0.45,
):
    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

    primary_tfl = []
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * strong_ratio),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
    if color_jitter is not None and color_aug:
        # color jitter is enabled when not using AA
        flip_and_color_jitter = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        secondary_tfl += flip_and_color_jitter

    if interpolation == "random":
        interpolation = (Image.BILINEAR, Image.BICUBIC)
    else:
        interpolation = _pil_interp(interpolation)
    final_tfl = [
        transforms.RandomResizedCrop(
            size=img_size, scale=scale, ratio=ratio, interpolation=Image.BICUBIC
        )
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [transforms.ToTensor()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
            )
        )
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


class DataAugmentation(object):
    """
    implement multi-crop data augmentation.
    --global_crops_scale: scale range of the 224-sized cropped image before resizing
    --local_crops_scale: scale range of the 96-sized cropped image before resizing
    --local_crops_number: Number of small local views to generate
    --prob: when we use strong augmentation and weak augmentation, the ratio of images to
        be cropped with strong augmentation
    --vanilla_weak_augmentation: whether we use the same augmentation in DINO, namely
        only using weak augmentation
    --color_aug: after AutoAugment, whether we further perform color augmentation
    --local_crop_size: the small crop size
    --timm_auto_augment_par: the parameters for the AutoAugment used in DeiT
    --strong_ratio: the ratio of image augmentation for the AutoAugment used in DeiT
    --re_prob: the re-prob parameter of image augmentation for the AutoAugment used in DeiT
    --use_prefetcher: whether we use prefetcher which can accerelate the training speed
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        prob=0.5,
        vanilla_weak_augmentation=False,
        color_aug=False,
        local_crop_size=[96],
        timm_auto_augment_par="rand-m9-mstd0.5-inc1",
        strong_ratio=0.45,
        re_prob=0.25,
        use_prefetcher=False,
    ):

        ## propability to perform strong augmentation
        self.prob = prob
        ## whether we use the commonly used augmentations, e.g. DINO or MoCo-V3
        self.vanilla_weak_augmentation = vanilla_weak_augmentation

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        if use_prefetcher:
            normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        ##====== build augmentation of global crops, i.e. 224-sized image crops =========
        # first global crop, always weak augmentation
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )

        # second global crop, always weak augmentation
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )

        # strong augmentation, maybe used if we need to perform strong augmentation
        self.global_transfo3 = strong_transforms(
            img_size=224,
            scale=global_crops_scale,
            ratio=(0.75, 1.3333333333333333),
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment=timm_auto_augment_par,  # 'rand-m9-mstd0.5-inc1'
            interpolation="random",
            use_prefetcher=use_prefetcher,  # True
            mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
            std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
            re_prob=re_prob,  # 0.25
            re_mode="pixel",
            re_count=1,
            re_num_splits=0,
            color_aug=color_aug,
            strong_ratio=strong_ratio,
        )

        ##====== build augmentation of local crops, i.e. 96-sized image crops =========
        self.local_crops_number = (
            local_crops_number  # transformation for the local small crops
        )
        assert local_crop_size[0] == 96
        # weak augmentation, maybe used if we need to perform weak augmentation
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size[0],
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )
        # strong augmentation, maybe used if we need to perform strong augmentation
        self.local_transfo2 = strong_transforms(
            img_size=local_crop_size[0],  # (224, 224)
            scale=local_crops_scale,  # (0.08, 1.0)
            ratio=(0.75, 1.3333333333333333),  # (0.75, 1.3333333333333333)
            hflip=0.5,  # 0.5
            vflip=0.0,  # 0.0
            color_jitter=0.4,  # 0.4
            auto_augment=timm_auto_augment_par,  # 'rand-m9-mstd0.5-inc1'
            interpolation="random",  # 'random'
            use_prefetcher=use_prefetcher,  # True
            mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
            std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
            re_prob=re_prob,  # 0.25
            re_mode="pixel",  # 'pixel'
            re_count=1,  # 1
            re_num_splits=0,  # 0
            color_aug=color_aug,
            strong_ratio=strong_ratio,
        )

    def __call__(self, image):
        """
        implement multi-crop data augmentation. Generate two 224-sized +
        "local_crops_number" 96-sized images
        """
        crops = []
        ##====== images to be fed into teacher, two 224-sized =========
        img1 = self.global_transfo1(image)
        img2 = self.global_transfo2(image)
        crops.append(img1)
        crops.append(img2)

        ##====== images to be fed into student, two 224-sized + "local_crops_number" 96-sized =========
        # first to generate two 224-sized
        # this weak_flag indicates whether the current image is weakly augmented.
        # For local group supervision, we only use weakly augmented images of size 224 to
        # update the memory for local-group aggregation.
        weak_flag = False

        if self.vanilla_weak_augmentation is True:
            ## directly copy the images of weak augmentation
            crops.append(copy.deepcopy(img1))
            crops.append(copy.deepcopy(img2))
            weak_flag = True
        elif self.prob < 1.0 and random.random() > self.prob:
            ## whether perform strong augmentation
            crops.append(self.global_transfo3(image))
            crops.append(self.global_transfo3(image))
        else:
            ## perform weak augmentation
            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image))
            weak_flag = True

        # then to generate "local_crops_number" 96-sized
        for _ in range(self.local_crops_number):
            if self.prob < 1.0 and random.random() > self.prob:
                ## whether perform strong augmentation
                crops.append(self.local_transfo2(image))
            else:
                ## perform weak augmentation
                crops.append(self.local_transfo(image))

        return crops, weak_flag


def get_dataset(args):
    """
    build a multi-crop data augmentation and a dataset/dataloader
    """
    ## preparing augmentations, including weak and strong augmentations
    transform = DataAugmentation(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        vanilla_weak_augmentation=args.vanilla_weak_augmentation,
        prob=args.prob,
        color_aug=args.color_aug,
        local_crop_size=args.size_crops,
        timm_auto_augment_par=args.timm_auto_augment_par,
        strong_ratio=args.strong_ratio,
        re_prob=args.re_prob,
        use_prefetcher=args.use_prefetcher,
    )

    ## For debug mode, we only load the first two classes to reduce data reading time.
    ## otherwise, we load all training data for pretraining.
    class_num = 2 if args.debug else 1000
    dataset = ImageFolder(args.data_path, transform=transform, class_num=class_num)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader


class data_prefetcher:
    """
    implement data prefetcher. we perform some augmentation on GPUs intead of CPUs
    --loader: a data loader
    --fp16: whether we use fp16, if yes, we need to tranform the data to be fp16
    """

    def __init__(self, loader, fp16=True):
        self.loader = iter(loader)
        self.fp16 = fp16
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.preload()

    def preload(self):
        """
        preload the next minibatch of data
        """
        try:
            self.multi_crops, self.weak_flag = next(self.loader)
        except StopIteration:
            self.multi_crops, self.weak_flag = None, None
            return

        with torch.cuda.stream(self.stream):
            for i in range(len(self.multi_crops)):
                self.multi_crops[i] = self.multi_crops[i].cuda(non_blocking=True)
                if self.fp16:
                    self.multi_crops[i] = (
                        self.multi_crops[i].half().sub_(self.mean).div_(self.std)
                    )
                else:
                    self.multi_crops[i] = (
                        self.multi_crops[i].float().sub_(self.mean).div_(self.std)
                    )

    def next(self):
        """
        load the next minibatch of data
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        multi_crops, weak_flags = self.multi_crops, self.weak_flag
        self.preload()
        return multi_crops, weak_flags
