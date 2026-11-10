import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_instancenorm,
    convert_dim_to_conv_op,
    maybe_convert_scalar_to_list,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He


class MedNeXtBlock(nn.Module):
    """
    3D ConvNeXt-style Block for MedNeXt.
    Depthwise 3D Conv -> Group/Instance Norm -> Inverted Bottleneck GELU -> Pointwise Conv -> Residual.
    """
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3, do_res=True, norm_type='group'):
        super().__init__()
        self.do_res = do_res
        if isinstance(kernel_size, (list, tuple)):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=True
        )
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        else:
            self.norm = nn.InstanceNorm3d(in_channels, affine=True)

        self.conv2 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, bias=True)

        if do_res:
            if in_channels != out_channels:
                self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        if self.do_res:
            x = x + self.res_conv(res)
        return x


class DynamicMedNeXtDownBlock(nn.Module):
    """Downsampling block adapting to arbitrary pooling strides [sz, sy, sx] from plans."""
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, exp_r=4):
        super().__init__()
        stride_tuple = tuple(stride)
        self.downsample = nn.Conv3d(
            in_channels, out_channels, kernel_size=stride_tuple,
            stride=stride_tuple, padding=0, bias=True
        )
        self.block = MedNeXtBlock(out_channels, out_channels, exp_r=exp_r, kernel_size=kernel_size)

    def forward(self, x):
        x = self.downsample(x)
        x = self.block(x)
        return x


class DynamicMedNeXtUpBlock(nn.Module):
    """Upsampling block adapting to arbitrary pooling strides [sz, sy, sx] from plans."""
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, exp_r=4):
        super().__init__()
        stride_tuple = tuple(stride)
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=stride_tuple,
            stride=stride_tuple, padding=0, bias=True
        )
        self.block = MedNeXtBlock(out_channels, out_channels, exp_r=exp_r, kernel_size=kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        return x


class DynamicMedNeXt(nn.Module):
    """
    Dynamic MedNeXt architecture matching exact nnU-Net plans (n_stages, strides, features).
    """
    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        kernel_sizes: List[List[int]],
        strides: List[List[int]],
        num_classes: int,
        exp_r: int = 4,
        deep_supervision: bool = True,
        block_counts: List[int] = None,
        **kwargs
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_stages = n_stages

        if block_counts is None:
            # Default MedNeXt-L block scaling adapted to n_stages
            block_counts = [2] * n_stages

        # Stem block (stage 0)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features_per_stage[0], kernel_size=1, bias=False),
            *[MedNeXtBlock(features_per_stage[0], features_per_stage[0], exp_r=exp_r, kernel_size=kernel_sizes[0])
              for _ in range(block_counts[0])]
        )

        # Encoder downsampling stages (stages 1 to n_stages-1)
        self.down_blocks = nn.ModuleList()
        self.enc_stages = nn.ModuleList()
        for s in range(1, n_stages):
            down = DynamicMedNeXtDownBlock(
                features_per_stage[s - 1], features_per_stage[s],
                stride=strides[s], kernel_size=kernel_sizes[s], exp_r=exp_r
            )
            self.down_blocks.append(down)
            stage_blocks = nn.Sequential(*[
                MedNeXtBlock(features_per_stage[s], features_per_stage[s], exp_r=exp_r, kernel_size=kernel_sizes[s])
                for _ in range(block_counts[s] - 1)
            ])
            self.enc_stages.append(stage_blocks)

        # Decoder upsampling stages
        self.up_blocks = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        for s in range(n_stages - 2, -1, -1):
            up = DynamicMedNeXtUpBlock(
                features_per_stage[s + 1], features_per_stage[s],
                stride=strides[s + 1], kernel_size=kernel_sizes[s], exp_r=exp_r
            )
            self.up_blocks.append(up)
            dec_conv = nn.Conv3d(features_per_stage[s] * 2, features_per_stage[s], kernel_size=1, bias=False)
            dec_block = MedNeXtBlock(features_per_stage[s], features_per_stage[s], exp_r=exp_r, kernel_size=kernel_sizes[s])
            self.dec_stages.append(nn.Sequential(dec_conv, dec_block))

            if self.deep_supervision and s > 0:
                self.ds_heads.append(nn.Conv3d(features_per_stage[s], num_classes, kernel_size=1))

        self.final_head = nn.Conv3d(features_per_stage[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)

        for down, stage in zip(self.down_blocks, self.enc_stages):
            x = down(x)
            x = stage(x)
            skips.append(x)

        ds_outputs = []
        for i, (up, dec) in enumerate(zip(self.up_blocks, self.dec_stages)):
            skip = skips[-(i + 2)]
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            # Collect deep supervision outputs for stages > 0
            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))

        out = self.final_head(x)

        if self.deep_supervision and self.training:
            # Returns [highest_res, ds_stage_N-1, ..., ds_stage_1] matching nnU-Net loss order
            return [out] + ds_outputs[::-1]
        return out


def get_mednext_3d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True
):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads

    features_per_stage = [
        min(configuration_manager.UNet_base_num_features * 2 ** i, configuration_manager.unet_max_num_features)
        for i in range(num_stages)
    ]

    model = DynamicMedNeXt(
        in_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=num_classes,
        exp_r=4,
        deep_supervision=deep_supervision,
    )
    model.apply(InitWeights_He(1e-2))
    return model
