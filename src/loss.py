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
functions for building multi-granular losses.
"""
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils import concat_all_gather


class InfoNCELoss(nn.Module):
    """
    vanilla infoNCEloss.
    --ncrops: how many crops are used in student networks
    --dim: feature dimension in queue determinted by output dimention of student network
    --queue_size: queue size
    --temperature: temperature parameter for infoNCEloss
    """

    def __init__(self, ncrops, dim=256, queue_size=65536, temperature=0.2):
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature

        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.ncrops = ncrops

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        queue update
        """
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            keys_t = keys.T
            queue_remaining_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys_t[:, :queue_remaining_size]
            self.queue[:, : batch_size - queue_remaining_size] = keys_t[
                :, queue_remaining_size:
            ]

            ptr = batch_size - queue_remaining_size  # move pointer

        self.queue_ptr[0] = ptr

    # student_output, teacher_output
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        preds = student_output.chunk(self.ncrops)
        targets = teacher_output.detach().chunk(2)
        small_crop_loss, large_crop_loss = 0, 0
        small_loss_terms, large_loss_terms = 0, 0
        queue_feat = self.queue.clone().detach()

        for t_idx, targ in enumerate(targets):
            for p_idx, pred in enumerate(preds):
                if t_idx == p_idx:
                    continue
                # positive logits: Nx1
                l_pos = torch.einsum("nc,nc->n", [pred, targ]).unsqueeze(-1)
                # negative logits: NxK
                l_neg = torch.einsum("nc,ck->nk", [pred, queue_feat])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
                # apply temperature
                logits /= self.temperature
                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
                    logits.device
                )
                loss = self.CrossEntropyLoss(logits, labels)
                if p_idx < 2:  ## large crop loss, namely loss on 224-sized images
                    large_crop_loss += loss
                    large_loss_terms += 1
                else:  ## small crop loss, namely loss on 96-sized images
                    small_crop_loss += loss
                    small_loss_terms += 1
            # dequeue and enqueue
            self._dequeue_and_enqueue(targ)

        large_crop_loss /= large_loss_terms
        small_crop_loss /= small_loss_terms
        loss = 0.5 * (large_crop_loss + small_crop_loss)
        return loss


class ClusteringLoss(nn.Module):
    """
    Clustering loss which is very simialr to the one in DINO
    --out_dim: center dimension determinted by output dimention of student network
    --ncrops: how many crops are used in student networks
    --warmup_teacher_temp: Initial value for the teacher temperature
    --teacher_temp: Final value (after linear warmup) of the teacher temperature
    --warmup_teacher_temp_epochs: Number of warmup epochs for the teacher temperature
    --nepochs: total training epoch
    --student_temp: temperature parameter in student output
    --center_momentum:  EMA parameter for center update
    """

    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        loss_large_crop, loss_small_crop = 0.0, 0.0
        loss_terms_large_crop, loss_terms_small_crop = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(
                    -q * F.log_softmax(student_out[v], dim=-1), dim=-1
                ).mean()
                if v < 2:
                    loss_large_crop += loss
                    loss_terms_large_crop += 1
                else:
                    loss_small_crop += loss
                    loss_terms_small_crop += 1

        self.update_center(teacher_output)
        loss_large_crop /= loss_terms_large_crop
        loss_small_crop /= loss_terms_small_crop
        total_loss = 0.5 * (loss_large_crop + loss_small_crop)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=False)
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def get_multi_granular_loss(args):
    """
    build the multi-granular loss
    """
    all_losses, all_weights = {}, {}

    ## build the instance discrimination loss
    instance_supervision_loss = InfoNCELoss(
        args.local_crops_number + 2,
        dim=args.instance_out_dim,
        queue_size=args.instance_queue_size,
        temperature=args.instance_temp,
    ).cuda()
    all_losses["instance-sup."] = instance_supervision_loss
    all_weights["instance-sup."] = args.loss_weights[0]

    ## build the local group discrimination loss
    local_group_supervision = InfoNCELoss(
        args.local_crops_number + 2,
        dim=args.local_group_out_dim,
        queue_size=args.local_group_queue_size,
        temperature=args.local_group_temp,
    ).cuda()
    all_losses["local-group-sup."] = local_group_supervision
    all_weights["local-group-sup."] = args.loss_weights[1]

    ## build the group discrimination loss
    group_loss = ClusteringLoss(
        args.group_out_dim,
        args.local_crops_number
        + 2,  # total number of crops = 2 global crops + local_crops_number
        args.group_warmup_teacher_temp,
        args.group_teacher_temp,
        args.group_warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.group_student_temp,
        center_momentum=0.9,
    ).cuda()
    all_losses["group-sup."] = group_loss
    all_weights["group-sup."] = args.loss_weights[2]
    return all_losses, all_weights
