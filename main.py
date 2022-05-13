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
Mugs training code
"""
import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models as torchvision_models

import utils
from src.loss import get_multi_granular_loss
from src.model import get_model
from src.multicropdataset import data_prefetcher, get_dataset
from src.optimizer import cancel_gradients_last_layer, get_optimizer, clip_gradients

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("Mugs", add_help=False)

    ##======== Model parameters ============
    parser.add_argument(
        "--arch",
        type=str,
        default="vit_small",
        choices=["vit_small", "vit_base", "vit_large"],
        help="""Name of architecture to train.""",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )

    ##======== Training/Optimization parameters ============
    parser.add_argument(
        "--momentum_teacher",
        type=float,
        default=0.996,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with
        cosine schedule. We recommend setting a higher value with small batches: for
        example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=False,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training
        with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.2,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=64,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs of training."
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="""Number of epochs for the linear learning-rate warm up.=""",
    )
    parser.add_argument(
        "--freeze_last_layer",
        type=int,
        default=1,
        help="""Number of epochs during
        which we keep the output layer fixed for the group supervision loss. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0008,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--patch_embed_lr_mult",
        type=float,
        default=0.2,
        help="""For patch
        embedding layer, its learning rate is lr * patch_embed_lr_mult (<1.0) in most case, which
        stables training and also slightly improve the performance.""",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw
        with ViTs.""",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="""stochastic depth rate"""
    )

    ##========  Multi-granular supervisions (instance/local-group/group supervisions) ==========
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="""three loss weights for instance, local-group, group supervision losses in turn""",
    )

    parser.add_argument(
        "--use_bn_in_head",
        type=utils.bool_flag,
        default=False,
        help="Whether to use batch normalizations in the three projection heads (Default: False)",
    )
    parser.add_argument(
        "--norm_before_pred",
        type=utils.bool_flag,
        default=True,
        help="""Whether to use batch normalizations after projection heads (namely before
        prediction heads) in instance and local-group supervisions. (Default: False)""",
    )

    # parameters for instance discrimination supervision
    parser.add_argument(
        "--instance_out_dim",
        type=int,
        default=256,
        help="""output dimention in the projection and prediction heads.""",
    )
    parser.add_argument(
        "--instance_queue_size",
        type=int,
        default=65536,
        help="""the queue size of the memory to store the negative keys.""",
    )
    parser.add_argument(
        "--instance_temp",
        type=float,
        default=0.2,
        help="""the temperature parameters for the infoNCE loss in instance supervision.""",
    )

    # parameters for local-group discrimination supervision
    parser.add_argument(
        "--local_group_out_dim",
        type=int,
        default=256,
        help="""output dimention in the projection and prediction heads.""",
    )
    parser.add_argument(
        "--local_group_knn_top_n",
        type=int,
        default=8,
        help="how many neighbors we use to aggregate for a local-group",
    )
    parser.add_argument(
        "--local_group_queue_size",
        type=int,
        default=65536,
        help="""the queue sizes of the memory to store the negative keys for infoNCE loss and
        another memory size to store the weak augmentated samples for local-group aggregation.""",
    )
    parser.add_argument(
        "--local_group_temp",
        type=float,
        default=0.2,
        help="""the temperature parameters for the infoNCE loss in instance supervision.""",
    )

    ## parameters for group discrimination supervision
    parser.add_argument(
        "--group_out_dim",
        type=int,
        default=65536,
        help="""output dimention in the prediction heads.""",
    )
    parser.add_argument(
        "--group_bottleneck_dim",
        type=float,
        default=256,
        help="""head bottleneck dimention in the prediction heads.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not to weight normalize the last layer of the group supervision head.
        Not normalizing leads to better performance but can make the training unstable. We
        typically set this paramater to False with vit_small and True with vit_base and vit_large.""",
    )

    parser.add_argument(
        "--group_student_temp",
        type=float,
        default=0.1,
        help="""the temperature parameters for the clustering loss in student output.""",
    )
    parser.add_argument(
        "--group_warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--group_teacher_temp",
        default=0.04,
        type=float,
        help="""Final value
        (after linear warmup) of the teacher temperature. For most experiments, anything above
        0.07 is unstable. We recommend starting with the default value of 0.04 and increase
        this slightly if needed.""",
    )
    parser.add_argument(
        "--group_warmup_teacher_temp_epochs",
        default=0,
        type=int,
        help="""Number of warmup epochs for the teacher temperature (Default: 30).""",
    )

    ##======== augmentation parameters  ============
    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.25, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=10,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )
    # strong augmentation parameters
    parser.add_argument(
        "--timm_auto_augment_par",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        help="""the parameters for the AutoAugment used in DeiT.""",
    )
    parser.add_argument(
        "--color_aug",
        type=utils.bool_flag,
        default=False,
        help="""after AutoAugment, whether we further perform color augmentation. (Default: False).""",
    )
    parser.add_argument(
        "--size_crops",
        type=int,
        default=[96],
        nargs="+",
        help="""the small crop size. Note we use multi-crop strategy, namely two 224-sized crops +
        ten 96-sized crops. (Default: 96)""",
    )
    parser.add_argument(
        "--strong_ratio",
        type=float,
        default=0.45,
        help="""the ratio of image augmentation for the AutoAugment used in DeiT.""",
    )
    parser.add_argument(
        "--re_prob",
        type=float,
        default=0.25,
        help="""the re-prob parameter of image augmentation for the AutoAugment used in DeiT.""",
    )
    parser.add_argument(
        "--vanilla_weak_augmentation",
        type=utils.bool_flag,
        default=False,
        help="""Whether we use the same augmentation in DINO, namely only using weak augmentation.""",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="""When we use strong augmentation and weak augmentation, the ratio of images to
        be cropped with strong augmentation.""",
    )

    ##======== Misc ============
    parser.add_argument(
        "--data_path",
        default="/dataset/imageNet100_sicy/train/",
        type=str,
        help="""Please specify path to the ImageNet training data.""",
    )
    parser.add_argument(
        "--output_dir",
        default="./exp/",
        type=str,
        help="""Path to save logs and checkpoints.""",
    )
    parser.add_argument(
        "--saveckp_freq",
        default=50,
        type=int,
        help="""Save checkpoint every x epochs.""",
    )
    parser.add_argument("--seed", default=0, type=int, help="""Random seed.""")
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="""Number of data loading workers per GPU.""",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="""local rank for distrbuted training.""",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="""world size for distrbuted training.""",
    )

    parser.add_argument(
        "--use_prefetcher",
        type=utils.bool_flag,
        default=True,
        help="""whether we use prefetcher which can accerelate the training speed.""",
    )
    parser.add_argument(
        "--debug",
        type=utils.bool_flag,
        default=False,
        help="""whether we debug. if yes, we only load small fraction of training data to reduce data reading time.""",
    )
    parser.add_argument(
        "--ddpjob",
        default=False,
        type=utils.bool_flag,
        help="""whether we use ddp job. We suggest to use it for distributed training. For single GPUs
        or Node, you can close it.""",
    )

    return parser


def train_mugs(args):
    """
    main training code for Mugs, including building dataloader, models, losses, optimizers, etc
    """
    ##======== prepare logger for more detailed logs ============
    logger = utils.get_logger(args.output_dir + "/train.log")
    logger.info(args)
    if args.output_dir and utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    ##======== initilize distribution ============
    if args.ddpjob is True:
        utils.init_distributed_ddpjob(args)
    else:
        utils.init_distributed_mode(args)

    ##======== fix seed for reproduce ============
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True
    cudnn.deterministic = True

    ##======== get the training dataset/loader ============
    data_loader = get_dataset(args)
    logger.info(f"Data loaded: there are {len(data_loader.dataset)} images.")

    ##====== build  student and teacher networks (vit_small, vit_base, vit_large) =========
    student, teacher, student_mem, teacher_mem = get_model(args)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    student_mem, teacher_mem = student_mem.cuda(), teacher_mem.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    ##======== get  multi granular losses and their loss weights ============
    all_losses, all_weights = get_multi_granular_loss(args)

    ##======== preparing optimizer ============
    optimizer, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule = get_optimizer(
        student, len(data_loader), args
    )

    ##======== optionally resume training ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        student_mem=student_mem,
        teacher_mem=teacher_mem,
        **all_losses,
    )
    start_epoch = to_restore["epoch"]

    ##======== Starting Mugs training ============
    logger.info("Starting Mugs training !")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        data_loader.sampler.set_epoch(epoch)

        ##======== training one epoch of Mugs ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            all_losses,
            all_weights,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            student_mem,
            teacher_mem,
            logger,
            args,
        )

        ##======== save model checkpoint ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "student_mem": student_mem.state_dict()
            if student_mem is not None
            else None,
            "teacher_mem": teacher_mem.state_dict()
            if teacher_mem is not None
            else None,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        granular_loss_dicts = {}
        for name, loss in all_losses.items():
            granular_loss_dicts[name] = loss.state_dict()
        save_dict.update(granular_loss_dicts)

        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )

        ##======== writing logs ============
        log_stats = {**{f"{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            t2 = time.time()
            log_results = ""
            for k, v in train_stats.items():
                log_results += "%s: %.6f, " % (k, v)
            logger.info(
                "%d-epoch: %s remaining time %.2f hours"
                % (epoch, log_results, (t2 - t1) * (args.epochs - epoch) / 3600.0)
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    all_losses,
    all_weights,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    student_mem,
    teacher_mem,
    logger,
    args,
):
    """
    main training code for each epoch
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    prefetcher = data_prefetcher(data_loader, fp16=(fp16_scaler is not None))
    images, weak_aug_flags = prefetcher.next()
    epoch_it = 0
    while images is not None:
        #  Step 1. update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + epoch_it  # global training iteration
        for _, param_group in enumerate(optimizer.param_groups):
            lr_mult = 1.0
            if "patch_embed" in param_group["name"]:
                lr_mult = args.patch_embed_lr_mult
            param_group["lr"] = lr_schedule[it] * lr_mult
            if param_group.get("apply_wd", True):  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        granular_losses = OrderedDict()
        total_loss = 0
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            ## Step 2. forward images into teacher and student to obtain the
            # features/superivisons for the three granular superivison losses
            (
                teacher_instance_target,
                teacher_local_group_target,
                teacher_group_target,
                teacher_memory_tokens,
            ) = teacher(
                images[:2],
                return_target=True,
                local_group_memory_inputs={"mem": teacher_mem},
            )

            (
                student_instance_target,
                student_local_group_target,
                student_group_target,
                student_memory_tokens,
            ) = student(
                images[2:],
                return_target=False,
                local_group_memory_inputs={"mem": student_mem},
            )

            ## Step 3. compute the three granular supervision losses, including instance,
            # local-group, group supervision losses
            weigts_sum, total_loss, granular_losses = 0.0, 0.0, OrderedDict()
            # instance loss
            loss_cls, loss_weight = (
                all_losses["instance-sup."],
                all_weights["instance-sup."],
            )
            if loss_weight > 0:
                instance_loss = loss_cls(
                    student_instance_target, teacher_instance_target, epoch
                )
                weigts_sum, total_loss = (
                    weigts_sum + loss_weight,
                    total_loss + instance_loss,
                )
                granular_losses["instance-sup."] = instance_loss.item()

            # local group loss
            loss_cls, loss_weight = (
                all_losses["local-group-sup."],
                all_weights["local-group-sup."],
            )
            if loss_weight > 0:
                local_group_loss = loss_cls(
                    student_local_group_target, teacher_local_group_target, epoch
                )
                weigts_sum, total_loss = (
                    weigts_sum + loss_weight,
                    total_loss + local_group_loss,
                )
                granular_losses["local-group-sup."] = local_group_loss.item()

            # group loss
            loss_cls, loss_weight = all_losses["group-sup."], all_weights["group-sup."]
            if loss_weight > 0:
                group_loss = loss_cls(student_group_target, teacher_group_target, epoch)
                weigts_sum, total_loss = (
                    weigts_sum + loss_weight,
                    total_loss + group_loss,
                )
                granular_losses["group-sup."] = group_loss.item()

            # average loss
            total_loss /= weigts_sum

            ## ## Step 4. update the memory buffer for local-group supervision losses.
            # for student, we only update memory by the image of size 224 and weak augmentations
            student_features = (student_memory_tokens.chunk(2))[0]
            len_weak = student_mem._dequeue_and_enqueue(
                student_features, 
                weak_aug_flags, 
            )

            teacher_weak = (teacher_memory_tokens.chunk(2))[0]
            _ = teacher_mem._dequeue_and_enqueue(teacher_weak, None)

        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()), force=True)
            sys.exit(1)

        ## Step 5. student and teacher update
        # student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            total_loss.backward()
            if args.clip_grad:
                clip_grad = args.clip_grad
                if epoch > 100 and args.arch == "vit_large":
                    clip_grad = args.clip_grad / 10.0
                _ = clip_gradients(student, clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(total_loss).backward()
            if args.clip_grad:
                clip_grad = args.clip_grad
                if epoch > 100 and args.arch == "vit_large":
                    clip_grad = args.clip_grad /10.0
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                _ = clip_gradients(student, clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                student.module.backbone.parameters(),
                teacher_without_ddp.backbone.parameters(),
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher_without_ddp.instance_head is not None:
                for param_q, param_k in zip(
                    student.module.instance_head.parameters(),
                    teacher_without_ddp.instance_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher_without_ddp.local_group_head is not None:
                for param_q, param_k in zip(
                    student.module.local_group_head.parameters(),
                    teacher_without_ddp.local_group_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if teacher_without_ddp.group_head is not None:
                for param_q, param_k in zip(
                    student.module.group_head.parameters(),
                    teacher_without_ddp.group_head.parameters(),
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        ## Step 6. load images
        images, weak_aug_flags = prefetcher.next()
        epoch_it += 1

        ## Step 7. logging
        torch.cuda.synchronize()
        metric_logger.update(loss=total_loss.item())
        for loss_name, loss_value in granular_losses.items():
            metric_logger.update(**{loss_name: loss_value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if epoch_it % 500 == 0 and args.rank == 0:  # and epoch_it < 10:
            log_results = ""
            for _, loss_name in enumerate(all_losses):
                if all_weights[loss_name] > 0:
                    log_results += "%s: %.6f," % (
                        loss_name,
                        metric_logger.meters[loss_name].global_avg,
                    )
            logger.info(
                "%d-epoch (%d/%d): total loss %.6f, %s, lr %.4e, wd %.4e, weak aug. ratio %.1f"
                % (
                    epoch,
                    it,
                    len(data_loader),
                    metric_logger.meters["loss"].global_avg,
                    log_results,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[0]["weight_decay"],
                    len_weak / len(weak_aug_flags) / args.world_size,
                )
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mugs", parents=[get_args_parser()])
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_mugs(args)
