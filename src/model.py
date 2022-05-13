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
models and functions for building student and teacher networks for multi-granular losses.
"""
import torch
import torch.nn as nn

import src.vision_transformer as vits
from src.vision_transformer import trunc_normal_


class Instance_Superivsion_Head(nn.Module):
    """
    a class to implement Instance Superivsion Head
    --in_dim: input dimension of projection head
    --hidden_dim: hidden dimension of projection head
    --out_dim: ouput dimension of projection and prediction heads
    --pred_hidden_dim: hidden dimension of prediction head
    --nlayers: layer number of projection head. prediction head has nlayers-1 layer
    --proj_bn: whether we use batch normalization in projection head
    --pred_bn: whether we use batch normalization in prediction head
    --norm_before_pred:  whether we use normalization before prediction head
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=2048,
        out_dim=256,
        pred_hidden_dim=4096,
        nlayers=3,
        proj_bn=False,
        pred_bn=False,
        norm_before_pred=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.norm_before_pred = norm_before_pred

        self.projector = self._build_mlp(
            nlayers, in_dim, hidden_dim, out_dim, use_bn=proj_bn
        )

        self.apply(self._init_weights)

        self.predictor = None
        if pred_hidden_dim > 0:  # teacher no, student yes
            self.predictor = self._build_mlp(
                nlayers - 1, out_dim, pred_hidden_dim, out_dim, use_bn=pred_bn
            )

    def _init_weights(self, m):
        """
        initilize the parameters in network
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_mlp(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False):
        """
        build a mlp
        """
        mlp = []
        for layer in range(num_layers):
            dim1 = input_dim if layer == 0 else hidden_dim
            dim2 = output_dim if layer == num_layers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if layer < num_layers - 1:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.GELU())

        return nn.Sequential(*mlp)

    def forward(self, x, return_target=False):
        """
        forward the input through projection head for teacher and
        projection/prediction heads for student
        """
        feat = self.projector(x)

        if return_target:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
            return feat
        ## return prediction
        if self.norm_before_pred:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
        pred = self.predictor(feat)
        pred = nn.functional.normalize(pred, dim=-1, p=2)
        return pred


class Local_Group_Superivsion_Head(nn.Module):
    """
    a class to implement Local Group Superivsion Head which is the same as Instance Superivsion Head
    --in_dim: input dimension of projection head
    --hidden_dim: hidden dimension of projection head
    --out_dim: ouput dimension of projection and prediction heads
    --pred_hidden_dim: hidden dimension of prediction head
    --nlayers: layer number of projection head. prediction head has nlayers-1 layer
    --proj_bn: whether we use batch normalization in projection head
    --pred_bn: whether we use batch normalization in prediction head
    --norm_before_pred:  whether we use normalization before prediction head
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=2048,
        out_dim=256,
        pred_hidden_dim=4096,
        nlayers=3,
        proj_bn=False,
        pred_bn=False,
        norm_before_pred=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.norm_before_pred = norm_before_pred

        self.projector = self._build_mlp(
            nlayers, in_dim, hidden_dim, out_dim, use_bn=proj_bn
        )

        self.apply(self._init_weights)

        self.predictor = None
        if pred_hidden_dim > 0:  # teacher no, student yes
            self.predictor = self._build_mlp(
                nlayers - 1, out_dim, pred_hidden_dim, out_dim, use_bn=pred_bn
            )

    def _init_weights(self, m):
        """
        initilize the parameters in network
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_mlp(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False):
        """
        build a mlp
        """
        mlp = []
        for layer in range(num_layers):
            dim1 = input_dim if layer == 0 else hidden_dim
            dim2 = output_dim if layer == num_layers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if layer < num_layers - 1:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.GELU())

        return nn.Sequential(*mlp)

    def forward(self, x, return_target=False):
        """
        forward the input through projection head for teacher and
        projection/prediction heads for student
        """
        feat = self.projector(x)

        if return_target:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
            return feat
        ## return prediction
        if self.norm_before_pred:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
        pred = self.predictor(feat)
        pred = nn.functional.normalize(pred, dim=-1, p=2)
        return pred


class Group_Superivsion_Head(nn.Module):
    """
    a class to implement Local Group Superivsion Head which is the same as Instance Superivsion Head
    --in_dim: input dimension of projection head
    --hidden_dim: hidden dimension of projection head
    --out_dim: ouput dimension of projection and prediction heads
    --pred_hidden_dim: hidden dimension of prediction head
    --nlayers: layer number of projection head. prediction head has nlayers-1 layer
    --proj_bn: whether we use batch normalization in projection head
    --pred_bn: whether we use batch normalization in prediction head
    --norm_before_pred:  whether we use normalization before prediction head
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
        use_bn=False,
        norm_last_layer=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        self.projector = self._build_mlp(
            nlayers, in_dim, hidden_dim, bottleneck_dim, use_bn=use_bn
        )
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _build_mlp(self, num_layers, in_dim, hidden_dim, output_dim, use_bn=False):
        """
        build a mlp
        """
        if num_layers == 1:
            mlp = nn.Linear(in_dim, output_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            mlp = nn.Sequential(*layers)
        return mlp

    def _init_weights(self, m):
        """
        initilize the parameters in network
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        forward the input through the projection and last prediction layer
        """
        feat = self.projector(x)
        feat = nn.functional.normalize(feat, dim=-1, p=2)
        feat = self.last_layer(feat)
        return feat


class Block_mem(nn.Module):
    """
    a class to implement a memory block for local group supervision
    --dim: feature vector dimenstion in the memory
    --K: memory size
    --top_n: number for neighbors in local group supervision
    """

    def __init__(self, dim, K=2048, top_n=10):
        super().__init__()
        self.dim = dim
        self.K = K
        self.top_n = top_n
        # create the queue
        self.register_buffer("queue_q", torch.randn(K, dim))
        self.register_buffer("queue_k", torch.randn(K, dim))
        self.register_buffer("queue_v", torch.randn(K, dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, query, weak_aug_flags):
        """
        update memory queue
        """
        # import pdb
        # pdb.set_trace()
        len_weak = 0
        query = concat_all_gather(query)
        if weak_aug_flags is not None:
            weak_aug_flags = weak_aug_flags.cuda()
            weak_aug_flags = concat_all_gather(weak_aug_flags)
            idx_weak = torch.nonzero(weak_aug_flags)
            len_weak = len(idx_weak)
            if len_weak > 0:
                idx_weak = idx_weak.squeeze(-1) 
                query = query[idx_weak]
            else:
                return len_weak

        all_size = query.shape[0]
        ptr = int(self.queue_ptr)
        remaining_size = ptr + all_size - self.K
        if remaining_size <= 0:
            self.queue_q[ptr : ptr + all_size, :] = query
            self.queue_k[ptr : ptr + all_size, :] = query
            self.queue_v[ptr : ptr + all_size, :] = query
            ptr = ptr + all_size
            self.queue_ptr[0] = (ptr + all_size) % self.K
        else:
            self.queue_q[ptr : self.K, :] = query[0 : self.K - ptr, :]
            self.queue_k[ptr : self.K, :] = query[0 : self.K - ptr, :]
            self.queue_v[ptr : self.K, :] = query[0 : self.K - ptr, :]

            self.queue_q[0:remaining_size, :] = query[self.K - ptr :, :]
            self.queue_k[0:remaining_size, :] = query[self.K - ptr :, :]
            self.queue_v[0:remaining_size, :] = query[self.K - ptr :, :]
            self.queue_ptr[0] = remaining_size
        return len_weak

    @torch.no_grad()
    def _get_similarity_index(self, x):
        """
        compute the index of the top-n neighbors (key-value pair) in memory
        """
        x = nn.functional.normalize(x, dim=-1)
        queue_q = nn.functional.normalize(self.queue_q, dim=-1)

        cosine = x @ queue_q.T
        _, index = torch.topk(cosine, self.top_n, dim=-1)
        return index

    @torch.no_grad()
    def _get_similarity_samples(self, query, index=None):
        """
        compute top-n neighbors (key-value pair) in memory
        """
        if index is None:
            index = self._get_similarity_index(query)
        get_k = self.queue_k[index.view(-1)]
        get_v = self.queue_v[index.view(-1)]
        B, tn = index.shape
        get_k = get_k.view(B, tn, self.dim)
        get_v = get_v.view(B, tn, self.dim)
        return get_k, get_v

    def forward(self, query):
        """
        forward to find the top-n neighbors (key-value pair) in memory
        """
        get_k, get_v = self._get_similarity_samples(query)
        return get_k, get_v


class vit_mem(nn.Module):
    """
    a class to implement a memory for local group supervision
    --dim: feature vector dimenstion in the memory
    --K: memory size
    --top_n: number for neighbors in local group supervision
    """

    def __init__(self, dim, K=2048, top_n=10):
        super().__init__()
        self.block = Block_mem(dim, K, top_n)

    def _dequeue_and_enqueue(self, query, weak_aug_flags):
        """
        update memory queue
        """
        query = query.float()
        weak_num = self.block._dequeue_and_enqueue(query, weak_aug_flags)
        return weak_num

    def forward(self, query):
        """
        forward to find the top-n neighbors (key-value pair) in memory
        """
        query = query.float()
        get_k, get_v = self.block(query)
        return get_k, get_v


class Mugs_Wrapper(nn.Module):
    """
    a class to implement a student or teacher wrapper for mugs
    --backbone: the backnone of student/teacher, e.g. ViT-small
    --instance_head: head, including projection/prediction heads, for instance supervision
    --local_group_head: head, including projection/prediction heads, for local group supervision
    --group_head: projection head for group supervision
    """

    def __init__(self, backbone, instance_head, local_group_head, group_head):
        super(Mugs_Wrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.instance_head = instance_head
        self.local_group_head = local_group_head
        self.group_head = group_head

    def forward(self, x, return_target=False, local_group_memory_inputs=None):
        """
        forward input to get instance/local-group/group targets or predictions
        """
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        class_tokens = torch.empty(0).to(x[0].device)
        mean_patch_tokens = torch.empty(0).to(x[0].device)
        memory_class_tokens = torch.empty(0).to(x[0].device)
        for _, end_idx in enumerate(idx_crops):
            input = torch.cat(x[start_idx:end_idx])
            token_feat, memory_class_token_feat = self.backbone(
                input,
                return_all=True,
                local_group_memory_inputs=local_group_memory_inputs,
            )  # [[16, 197, 384], [16, 384]] teacher
            # [[16, 197, 384], [16, 384]] student  [[48, 37, 384], [48, 384]]

            class_token_feat = token_feat[
                :, 0
            ]  # class tokens in ViT, [16, 384] teacher [16, 384] student  [48, 384]
            class_tokens = torch.cat((class_tokens, class_token_feat))

            start_idx = end_idx

            if self.local_group_head is not None:
                memory_class_tokens = torch.cat(
                    (memory_class_tokens, memory_class_token_feat)
                )
                if input.shape[-1] == 224:
                    mean_patch_tokens = torch.cat(
                        (mean_patch_tokens, token_feat[:, 1:].mean(dim=1))
                    )

        ## target [16, 256] for teacher,  [64, 256] for student,
        instance_feat = (
            self.instance_head(class_tokens, return_target)
            if self.instance_head is not None
            else None
        )

        ## target [16, 256] for teacher,  [64, 256] for student
        local_group_feat = (
            self.local_group_head(memory_class_tokens, return_target)
            if self.local_group_head is not None
            else None
        )

        # target [16, 65536] for teacher, [64, 65536] for student
        group_feat = (
            self.group_head(class_tokens) if self.group_head is not None else None
        )
        return instance_feat, local_group_feat, group_feat, mean_patch_tokens.detach()


def get_model(args):
    """
    build a student or teacher for mugs, includeing backbone, instance/local-group/group heads,
    and memory buffer
    """
    ## backbone
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            num_relation_blocks=1,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size, num_relation_blocks=1
        )
        embed_dim = student.embed_dim
    else:
        assert f"Unknow architecture: {args.arch}"

    ## memory buffer for local-group loss
    student_mem = vit_mem(
        embed_dim, K=args.local_group_queue_size, top_n=args.local_group_knn_top_n
    )
    teacher_mem = vit_mem(
        embed_dim, K=args.local_group_queue_size, top_n=args.local_group_knn_top_n
    )

    ## multi-crop wrapper handles forward with inputs of different resolutions
    student_instance_head, student_local_group_head, student_group_head = (
        None,
        None,
        None,
    )
    teacher_instance_head, teacher_local_group_head, teacher_group_head = (
        None,
        None,
        None,
    )

    # instance head
    if args.loss_weights[0] > 0:
        student_instance_head = Instance_Superivsion_Head(
            in_dim=embed_dim,
            hidden_dim=2048,
            out_dim=args.instance_out_dim,
            pred_hidden_dim=4096,
            nlayers=3,
            proj_bn=args.use_bn_in_head,
            pred_bn=False,
            norm_before_pred=args.norm_before_pred,
        )
        teacher_instance_head = Instance_Superivsion_Head(
            in_dim=embed_dim,
            hidden_dim=2048,
            out_dim=args.instance_out_dim,
            pred_hidden_dim=0,
            nlayers=3,
            proj_bn=args.use_bn_in_head,
            pred_bn=False,
            norm_before_pred=args.norm_before_pred,
        )

    # local group head
    if args.loss_weights[1] > 0:
        student_local_group_head = Local_Group_Superivsion_Head(
            in_dim=embed_dim,
            hidden_dim=2048,
            out_dim=args.local_group_out_dim,
            pred_hidden_dim=4096,
            nlayers=3,
            proj_bn=args.use_bn_in_head,
            pred_bn=False,
            norm_before_pred=args.norm_before_pred,
        )
        teacher_local_group_head = Local_Group_Superivsion_Head(
            in_dim=embed_dim,
            hidden_dim=2048,
            out_dim=args.local_group_out_dim,
            pred_hidden_dim=0,
            nlayers=3,
            proj_bn=args.use_bn_in_head,
            pred_bn=False,
            norm_before_pred=args.norm_before_pred,
        )

    # group head
    if args.loss_weights[2] > 0:
        student_group_head = Group_Superivsion_Head(
            in_dim=embed_dim,
            out_dim=args.group_out_dim,
            hidden_dim=2048,
            bottleneck_dim=args.group_bottleneck_dim,
            nlayers=3,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher_group_head = Group_Superivsion_Head(
            in_dim=embed_dim,
            out_dim=args.group_out_dim,
            hidden_dim=2048,
            bottleneck_dim=args.group_bottleneck_dim,
            nlayers=3,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )

    # multi-crop wrapper
    student = Mugs_Wrapper(
        student, student_instance_head, student_local_group_head, student_group_head
    )

    teacher = Mugs_Wrapper(
        teacher, teacher_instance_head, teacher_local_group_head, teacher_group_head
    )

    return student, teacher, student_mem, teacher_mem


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
