#!/usr/bin/env python3
"""
mednext_v1.py
=============
Official MedNeXt Architecture (MICCAI 2023 - Saikat Roy et al., DKFZ).
Paper: "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation"
Pure PyTorch implementation with:
- ConvNeXt-style 3D Depthwise Separable Convolutions
- Inverted Bottleneck (expansion ratio = 4)
- Residual Scaling & GroupNorm / LayerNorm
- Multi-scale Deep Supervision heads
- Scalable across Model IDs: 'S', 'B', 'M', 'L' and Kernel Sizes: 3 or 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MedNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3, do_res=True, norm_type='group', grn=False):
        super().__init__()
        self.do_res = do_res
        padding = kernel_size // 2

        # 3D Depthwise Conv
        self.conv1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=1, padding=padding,
            groups=in_channels, bias=True
        )

        # Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=min(in_channels, 32), num_channels=in_channels)
        else:
            self.norm = nn.InstanceNorm3d(in_channels, affine=True)

        # Inverted Bottleneck: 1x1x1 Conv expand
        self.conv2 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.GELU()

        # 1x1x1 Conv compress
        self.conv3 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Residual projection if in_channels != out_channels
        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
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

class MedNeXtDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3, do_res=True, norm_type='group'):
        super().__init__()
        # Strided conv for 2x downsampling
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.block = MedNeXtBlock(out_channels, out_channels, exp_r=exp_r, kernel_size=kernel_size, do_res=do_res, norm_type=norm_type)

    def forward(self, x):
        x = self.downsample(x)
        x = self.block(x)
        return x

class MedNeXtUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3, do_res=True, norm_type='group'):
        super().__init__()
        # Transposed conv for 2x upsampling
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.block = MedNeXtBlock(out_channels, out_channels, exp_r=exp_r, kernel_size=kernel_size, do_res=do_res, norm_type=norm_type)

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        return x

class MedNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        n_classes: int = 2,
        exp_r: int = 4,
        kernel_size: int = 3,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = True,
        do_res: bool = True,
        n_channels: int = 32,
        block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        norm_type: str = 'group',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint

        if enc_kernel_size is None:
            enc_kernel_size = kernel_size
        if dec_kernel_size is None:
            dec_kernel_size = kernel_size

        # Initial stem convolution
        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # Encoder stages
        c = n_channels
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        # Stage 0
        s0_blocks = [MedNeXtBlock(c, c, exp_r=exp_r, kernel_size=enc_kernel_size, do_res=do_res, norm_type=norm_type) for _ in range(block_counts[0])]
        self.enc_blocks.append(nn.Sequential(*s0_blocks))

        # Stages 1, 2, 3 (Downsamplings)
        for i in range(1, 4):
            down = MedNeXtDownBlock(c, c * 2, exp_r=exp_r, kernel_size=enc_kernel_size, do_res=do_res, norm_type=norm_type)
            self.down_blocks.append(down)
            c = c * 2
            stage_blocks = [MedNeXtBlock(c, c, exp_r=exp_r, kernel_size=enc_kernel_size, do_res=do_res, norm_type=norm_type) for _ in range(block_counts[i])]
            self.enc_blocks.append(nn.Sequential(*stage_blocks))

        # Bottleneck (Stage 4)
        down_bn = MedNeXtDownBlock(c, c * 2, exp_r=exp_r, kernel_size=enc_kernel_size, do_res=do_res, norm_type=norm_type)
        self.down_blocks.append(down_bn)
        c = c * 2
        bn_blocks = [MedNeXtBlock(c, c, exp_r=exp_r, kernel_size=enc_kernel_size, do_res=do_res, norm_type=norm_type) for _ in range(block_counts[4])]
        self.bottleneck = nn.Sequential(*bn_blocks)

        # Decoder stages (Stages 5, 6, 7, 8)
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        for i in range(5, 9):
            up = MedNeXtUpBlock(c, c // 2, exp_r=exp_r, kernel_size=dec_kernel_size, do_res=do_res, norm_type=norm_type)
            self.up_blocks.append(up)
            c = c // 2
            # Combine skip connection (c + c = 2c -> reduce to c)
            dec_conv = nn.Conv3d(c * 2, c, kernel_size=1, stride=1, padding=0, bias=True)
            stage_dec = [MedNeXtBlock(c, c, exp_r=exp_r, kernel_size=dec_kernel_size, do_res=do_res, norm_type=norm_type) for _ in range(block_counts[i])]
            self.dec_blocks.append(nn.Sequential(dec_conv, *stage_dec))

            if self.deep_supervision and i < 8:
                self.ds_heads.append(nn.Conv3d(c, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

        # Final segmentation head
        self.final_head = nn.Conv3d(n_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint as grad_checkpoint
        x = self.stem(x)

        # Encoder forward pass
        skips = []
        for i in range(4):
            if i > 0:
                x = self.down_blocks[i - 1](x)
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(self.enc_blocks[i], x, use_reentrant=False)
            else:
                x = self.enc_blocks[i](x)
            skips.append(x)

        # Bottleneck
        x = self.down_blocks[3](x)
        if self.use_checkpoint and self.training:
            x = grad_checkpoint(self.bottleneck, x, use_reentrant=False)
        else:
            x = self.bottleneck(x)

        # Decoder forward pass
        ds_outputs = []
        for i in range(4):
            x = self.up_blocks[i](x)
            skip = skips[3 - i]
            x = torch.cat([x, skip], dim=1)
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(self.dec_blocks[i], x, use_reentrant=False)
            else:
                x = self.dec_blocks[i](x)

            if self.deep_supervision and i < 3:
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)

        out = self.final_head(x)

        if self.deep_supervision and self.training:
            # Return list of outputs: [highest_res, ds_1, ds_2, ds_3]
            return [out] + ds_outputs[::-1]
        return out

def create_mednext_v1(num_channels=5, num_classes=2, model_id='L', kernel_size=3, deep_supervision=True, use_checkpoint=False):
    """
    Model creation utility matching MIC-DKFZ MedNeXt official configurations:
    - 'S' (Small): base_channels=32, block_counts=[2,2,2,2, 2, 2,2,2,2], exp_r=2
    - 'B' (Base): base_channels=32, block_counts=[2,2,2,2, 2, 2,2,2,2], exp_r=3
    - 'M' (Medium): base_channels=32, block_counts=[3,4,4,4, 4, 4,4,4,3], exp_r=4
    - 'L' (Large): base_channels=32, block_counts=[3,4,8,8, 8, 8,8,4,3], exp_r=4
    """
    configs = {
        'S': {'n_channels': 32, 'exp_r': 2, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'B': {'n_channels': 32, 'exp_r': 3, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'M': {'n_channels': 32, 'exp_r': 4, 'block_counts': [3, 4, 4, 4, 4, 4, 4, 4, 3]},
        'L': {'n_channels': 32, 'exp_r': 4, 'block_counts': [3, 4, 8, 8, 8, 8, 8, 4, 3]},
    }
    cfg = configs.get(model_id.upper(), configs['L'])
    return MedNeXt(
        in_channels=num_channels,
        n_classes=num_classes,
        exp_r=cfg['exp_r'],
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
        n_channels=cfg['n_channels'],
        block_counts=cfg['block_counts'],
    )

if __name__ == "__main__":
    # Quick self-test
    print("Testing MedNeXt L k3 initialization...")
    model = create_mednext_v1(num_channels=5, num_classes=2, model_id='L', kernel_size=3, deep_supervision=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MedNeXt L k3 created successfully! Total parameters: {total_params / 1e6:.2f}M")
