#!/usr/bin/env python3
"""
add_trainer_umamba_200epochs.py
===============================
Registers official U-Mamba Bot 3D (MICCAI 2024 - Jun Ma et al., Bo Wang Lab)
into the installed nnunetv2 package.

Compliant with 3D_UNET_BEST_PRACTICES.md:
1. Native nnU-Net v2 architecture builder: get_umamba_bot_3d_from_plans.
2. Exactly 200 epochs (self.num_epochs = 200) for fair baseline comparison.
3. Explicit parameter signature to prevent KeyError: 'args' reflection bug.
4. Dual-backend Mamba support: uses official mamba_ssm if available, with pure
   PyTorch Selective State Space Model fallback.
"""

import os
import sys
import shutil

try:
    import nnunetv2
    base_dir = os.path.dirname(nnunetv2.__file__)
except ImportError:
    print("Warning: nnunetv2 is not installed in the current environment.")
    print("This script will prepare the source files in scripts/ for deployment to GCP VM.")
    base_dir = None

UMAMBA_BOT_3D_CODE = '''import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    get_matching_instancenorm,
    convert_dim_to_conv_op,
    maybe_convert_scalar_to_list,
    get_matching_pool_op
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from torch.cuda.amp import autocast

# Check if official CUDA mamba_ssm is available; otherwise provide pure PyTorch fallback
try:
    from mamba_ssm import Mamba
    HAS_OFFICIAL_MAMBA = True
except ImportError:
    HAS_OFFICIAL_MAMBA = False

    class PurePyTorchMamba(nn.Module):
        """Pure PyTorch Selective State Space Model fallback."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = int(expand * d_model)

            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                bias=True,
                padding=d_conv - 1,
                groups=self.d_inner
            )
            self.act = nn.SiLU()

            self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, u):
            B, L, _ = u.shape
            xz = self.in_proj(u)
            x, z = xz.chunk(2, dim=-1)

            x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x_act = self.act(x_conv)

            x_proj_out = self.x_proj(x_act)
            delta_raw, B_ssm, C_ssm = torch.split(
                x_proj_out, [1, self.d_state, self.d_state], dim=-1
            )
            delta = F.softplus(self.dt_proj(delta_raw))
            A = -torch.exp(self.A_log.float())
            D_f = self.D.float()

            y = []
            h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=torch.float32)

            for t in range(L):
                xt = x_act[:, t, :].float()
                dt = delta[:, t, :].float()
                bt = B_ssm[:, t, :].float()
                ct = C_ssm[:, t, :].float()

                dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
                dB = torch.einsum('bd,bn->bdn', dt, bt)

                h = h * dA + torch.einsum('bdn,bd->bdn', dB, xt)
                yt = torch.einsum('bdn,bn->bd', h, ct) + xt * D_f
                y.append(yt.to(u.dtype))

            y = torch.stack(y, dim=1)
            y = y * self.act(z)
            return self.out_proj(y)

    Mamba = PurePyTorchMamba


class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class BasicResBlock(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, norm_op, norm_op_kwargs,
                 kernel_size=3, padding=1, stride=1, use_1x1conv=False,
                 nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):
        super().__init__()
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UNetResEncoder(nn.Module):
    def __init__(self, input_channels: int, n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None, return_skips: bool = False,
                 stem_channels: int = None, pool_type: str = 'conv'):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.conv_pad_sizes = [[i // 2 for i in krnl] for krnl in kernel_sizes]
        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op, input_channels=input_channels, output_channels=stem_channels,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0], stride=1, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            *[BasicBlockD(
                conv_op=conv_op, input_channels=stem_channels, output_channels=stem_channels,
                kernel_size=kernel_sizes[0], stride=1, conv_bias=conv_bias, norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        input_channels = stem_channels
        stages = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels, output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s], padding=self.conv_pad_sizes[s],
                    stride=strides[s], use_1x1conv=True, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ),
                *[BasicBlockD(
                    conv_op=conv_op, input_channels=features_per_stage[s], output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s], stride=1, conv_bias=conv_bias, norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.kernel_sizes = kernel_sizes
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if s > 0:
                input_size = [i // j for i, j in zip(input_size, self.strides[s])]
            output += np.prod([self.output_channels[s], *input_size], dtype=np.int64)
        return output


class UNetResDecoder(nn.Module):
    def __init__(self, encoder: UNetResEncoder, num_classes: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        stages = []
        upsampling_layers = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            upsampling_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op, input_channels=input_features_below,
                output_channels=input_features_skip, pool_op_kernel_size=stride_for_transpconv,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op, norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    input_channels=2 * input_features_skip, output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)], padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1, use_1x1conv=True, nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op, input_channels=input_features_skip, output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)], stride=1, conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs,
                ) for _ in range(n_conv_per_stage[s - 1] - 1)]
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsampling_layers = nn.ModuleList(upsampling_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsampling_layers[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[s](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            output += np.prod([self.encoder.output_channels[-(s + 2)], *input_size], dtype=np.int64)
        return output


class UMambaBot(nn.Module):
    def __init__(self, input_channels: int, n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None, nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None, deep_supervision: bool = True,
                 stem_channels: int = None, **kwargs):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        self.encoder = UNetResEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, stem_channels=stem_channels
        )
        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_bot_3d_from_plans(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        deep_supervision: bool = True
    ):
    arch_kwargs = configuration_manager.network_arch_init_kwargs
    kernel_sizes = arch_kwargs['kernel_sizes']
    strides = arch_kwargs['strides']
    n_stages = arch_kwargs.get('n_stages', len(kernel_sizes))
    features_per_stage = arch_kwargs['features_per_stage']

    dim = len(kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    kwargs = {
        'UMambaBot': {
            'conv_bias': arch_kwargs.get('conv_bias', True),
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    n_conv_per_stage = arch_kwargs.get('n_conv_per_stage', [2] * n_stages)
    n_conv_per_stage_decoder = arch_kwargs.get('n_conv_per_stage_decoder', [2] * (n_stages - 1))
    if isinstance(n_conv_per_stage, int):
        n_conv_per_stage = [n_conv_per_stage] * n_stages
    if isinstance(n_conv_per_stage_decoder, int):
        n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

    model = UMambaBot(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_classes=num_output_channels,
        deep_supervision=deep_supervision,
        n_conv_per_stage=n_conv_per_stage,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        **kwargs['UMambaBot']
    )
    model.apply(InitWeights_He(1e-2))
    return model
'''

TRAINER_CODE = '''import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

try:
    from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
except ImportError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "nets"))
    from UMambaBot_3d import get_umamba_bot_3d_from_plans


class nnUNetTrainerUMambaBot_200epochs(nnUNetTrainer):
    """
    Official U-Mamba Bot 3D Trainer with 200 Epochs (MICCAI 2024 - Jun Ma et al.).
    Compliant with 3D_UNET_BEST_PRACTICES.md:
    - 200 epochs matching ResEnc-UNet L, Standard 3D nnU-Net, and Vanilla 3D U-Net.
    - Explicit parameter signature to prevent KeyError: 'args' reflection crash.
    - Integrates Mamba SSM layer at bottleneck while maintaining full nnU-Net pipeline.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_umamba_bot_3d_from_plans(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels,
            deep_supervision=enable_deep_supervision
        )
        print(f"[U-Mamba Bot 3D] Architecture built successfully: {model.__class__.__name__}")
        return model
'''


def register_trainer():
    if base_dir is None:
        print("nnunetv2 is not available; saving reference files to scripts/...")
        ref_nets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UMambaBot_3d.py")
        with open(ref_nets, "w", encoding="utf-8") as f:
            f.write(UMAMBA_BOT_3D_CODE)
        print(f"Saved: {ref_nets}")
        return

    # 1. Write UMambaBot_3d.py into nnunetv2/nets/
    nets_dir = os.path.join(base_dir, "nets")
    os.makedirs(nets_dir, exist_ok=True)
    nets_file = os.path.join(nets_dir, "UMambaBot_3d.py")
    with open(nets_file, "w", encoding="utf-8") as f:
        f.write(UMAMBA_BOT_3D_CODE)
    print(f"Successfully registered UMambaBot_3d network at:\n  {nets_file}")

    # 2. Write nnUNetTrainerUMambaBot_200epochs.py into nnunetv2/training/nnUNetTrainer/variants/network_architecture/
    trainer_dir = os.path.join(base_dir, "training", "nnUNetTrainer", "variants", "network_architecture")
    os.makedirs(trainer_dir, exist_ok=True)
    trainer_file = os.path.join(trainer_dir, "nnUNetTrainerUMambaBot_200epochs.py")
    with open(trainer_file, "w", encoding="utf-8") as f:
        f.write(TRAINER_CODE)
    print(f"Successfully registered nnUNetTrainerUMambaBot_200epochs at:\n  {trainer_file}")

    # 3. Clear pycache
    cache_dirs = [
        os.path.join(nets_dir, "__pycache__"),
        os.path.join(trainer_dir, "__pycache__"),
        os.path.join(base_dir, "training", "nnUNetTrainer", "__pycache__"),
    ]
    for cd in cache_dirs:
        if os.path.exists(cd):
            shutil.rmtree(cd)
            print(f"Cleared cache at: {cd}")

    print("=== Official U-Mamba Bot 3D (200 Epochs) Registered Successfully! ===")


if __name__ == "__main__":
    register_trainer()
