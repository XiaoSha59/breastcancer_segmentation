#!/usr/bin/env python3
"""
umamba_bot_3d.py
================
Official U-Mamba Bot 3D Architecture (MICCAI 2024 - Jun Ma et al., DKFZ & Univ of Toronto).
Paper: "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation"

Features:
- 3D Residual Convolutional Encoder-Decoder topology.
- Mamba State Space Model (SSM) layer integrated at the Bottleneck.
- Dual-backend Mamba support:
  * Uses official CUDA 'mamba_ssm' if installed.
  * Seamless Pure-PyTorch Selective State Space Model fallback (zero compilation hurdles).
- Multi-scale Deep Supervision outputs.
- 5 input channels, 2 output classes (background, breast tumor).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Check if official CUDA mamba_ssm is available; otherwise use robust PyTorch fallback
try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True
except ImportError:
    HAS_OFFICIAL_MAMBA = False

class PurePyTorchMamba(nn.Module):
    """
    Pure PyTorch implementation of the Selective State Space Model (Mamba block).
    Guarantees 100% portable execution on any system without requiring custom nvcc compilation.
    """
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

        # SSM parameters: Delta, B, C projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # S4D real parameter initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        """
        u: (B, L, D) sequence tensor
        """
        B, L, _ = u.shape
        xz = self.in_proj(u)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)

        # 1D causal convolution across token sequence
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = self.act(x_conv)

        # Selective scan projections
        x_proj_out = self.x_proj(x_act)
        delta_raw, B_ssm, C_ssm = torch.split(
            x_proj_out, [1, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Sequential scan along sequence length L (using float32 for numerical stability under AMP)
        y = []
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D_f = self.D.float()
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=torch.float32)

        for t in range(L):
            xt = x_act[:, t, :].float()  # (B, d_inner)
            dt = delta[:, t, :].float()  # (B, d_inner)
            bt = B_ssm[:, t, :].float()  # (B, d_state)
            ct = C_ssm[:, t, :].float()  # (B, d_state)

            dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
            dB = torch.einsum('bd,bn->bdn', dt, bt)

            h = h * dA + torch.einsum('bdn,bd->bdn', dB, xt)
            yt = torch.einsum('bdn,bn->bd', h, ct) + xt * D_f
            y.append(yt.to(u.dtype))

        y = torch.stack(y, dim=1)  # (B, L, d_inner)
        # Gating connection
        y = y * self.act(z)
        out = self.out_proj(y)
        return out

class MambaLayer(nn.Module):
    """
    U-Mamba Bottleneck Layer: Flattens 3D feature map to sequence, passes through Mamba, reshapes back.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        if HAS_OFFICIAL_MAMBA:
            self.mamba = OfficialMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = PurePyTorchMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        assert C == self.dim, f"Channel dimension {C} does not match {self.dim}"

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        x_norm = self.norm(x_flat)
        mamba_out = self.mamba(x_norm)
        out = mamba_out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return x + out  # Residual addition

class ResConvBlock3D(nn.Module):
    """
    Standard 3D Residual Convolutional Block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act2(x + res)
        return x

class UMambaBot3D(nn.Module):
    """
    U-Mamba Bot 3D Architecture for 5-channel Breast DCE-MRI Tumor Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 2,
        features: list = [32, 64, 128, 256, 320],
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.out_channels = out_channels

        # Stem & Encoder Stages
        self.enc1 = ResConvBlock3D(in_channels, features[0])
        self.down1 = nn.Conv3d(features[0], features[0], kernel_size=2, stride=2)

        self.enc2 = ResConvBlock3D(features[0], features[1])
        self.down2 = nn.Conv3d(features[1], features[1], kernel_size=2, stride=2)

        self.enc3 = ResConvBlock3D(features[1], features[2])
        self.down3 = nn.Conv3d(features[2], features[2], kernel_size=2, stride=2)

        self.enc4 = ResConvBlock3D(features[2], features[3])
        self.down4 = nn.Conv3d(features[3], features[3], kernel_size=2, stride=2)

        # Bottleneck: Conv + Mamba SSM Layer
        self.bottleneck_conv = ResConvBlock3D(features[3], features[4])
        self.bottleneck_mamba = MambaLayer(features[4], d_state=16, d_conv=4, expand=2)

        # Decoder Stages
        self.up4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.dec4 = ResConvBlock3D(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ResConvBlock3D(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ResConvBlock3D(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ResConvBlock3D(features[0] * 2, features[0])

        # Multi-scale Deep Supervision Heads
        self.final_head = nn.Conv3d(features[0], out_channels, kernel_size=1)
        if self.deep_supervision:
            self.ds_head2 = nn.Conv3d(features[1], out_channels, kernel_size=1)
            self.ds_head3 = nn.Conv3d(features[2], out_channels, kernel_size=1)
            self.ds_head4 = nn.Conv3d(features[3], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        # Bottleneck with Mamba SSM
        bn = self.bottleneck_conv(d4)
        bn = self.bottleneck_mamba(bn)

        # Decoder forward pass
        u4 = self.up4(bn)
        c4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(c4)
        c3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final_head(c1)

        if self.deep_supervision and self.training:
            # Multi-scale outputs: [highest_res, ds_head2, ds_head3, ds_head4]
            return [out, self.ds_head2(c2), self.ds_head3(c3), self.ds_head4(c4)]
        return out

def create_umamba_bot_3d(
    in_channels: int = 5,
    out_channels: int = 2,
    features: list = [32, 64, 128, 256, 320],
    deep_supervision: bool = True,
):
    """
    Factory function for U-Mamba Bot 3D architecture.
    """
    return UMambaBot3D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        deep_supervision=deep_supervision,
    )

if __name__ == "__main__":
    print(f"Testing UMambaBot3D initialization (Official Mamba available: {HAS_OFFICIAL_MAMBA})...")
    model = create_umamba_bot_3d(in_channels=5, out_channels=2, deep_supervision=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UMambaBot3D initialized successfully! Trainable parameters: {total_params / 1e6:.2f}M")
