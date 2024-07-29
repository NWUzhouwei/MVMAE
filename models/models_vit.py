# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from models.utils import create_pointnet_components
from torch.nn import functional as F


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    # blocks = ((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None))
    def __init__(self,num_classes=40, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
  
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm


        
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=((64, 1, 30), (128, 2, 15), (512, 1, None), (1024, 1, None)), in_channels=6, normalize=False,
            width_multiplier=1, voxel_resolution_multiplier=1,model='PVTConv'
        )
        self.point_features = nn.ModuleList(layers)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(channels_point + concat_channels_point + channels_point, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024))
        self.linear1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, 768)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(1536, 768)

    def forward_features(self, data):
        # x = data[0]
        x = data

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]






        # features = data[1].permute(0, 2, 1)
        # out_features_list = []
        # coords = features[:,:3,:]
        # num_points, batch_size = features.size(-1), features.size(0)
        # for i in range(len(self.point_features)):
        #     features, _ = self.point_features[i]((features, coords))
        #     out_features_list.append(features)
        # out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        # out_features_list.append(
        #     features.mean(dim=-1, keepdim=True).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_points))

        # features = torch.cat(out_features_list, dim=1)
        # features = F.leaky_relu(self.conv_fuse(features))
        # features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        # features = F.leaky_relu(self.bn1(self.linear1(features)))
        # features = self.dp1(features)
        # features = F.leaky_relu(self.bn2(self.linear2(features)))
        # features = self.dp2(features)
        # outcome2 = self.linear3(features)

        # outcome = torch.cat((outcome,outcome2),dim=1)
        # outcome = self.linear4(outcome)



        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model