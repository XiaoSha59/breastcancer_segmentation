#!/usr/bin/env python3
"""
primus_v2.py
============
Official PrimusV2 Architecture Wrapper for 3D Breast Cancer DCE-MRI Segmentation.
Based on: "Primus: Enforcing Attention Usage for 3D Medical Image Segmentation"
Authors: Tassilo Wald, Saikat Roy, Fabian Isensee, Klaus Maier-Hein (MIC-DKFZ, 2024).

Features:
- Pure 3D Transformer-Centric Backbone (EVA ViT with 3D Axial RoPE)
- Iterative Residual Tokenizer (preserves micro-details and tumor contours)
- SwiGLU MLP + LayerScale + Post-Attention Normalization
- Configured specifically for 5-channel DCE-MRI inputs [I0, Ie, Il, dIe, dIl]
"""

from typing import Tuple
import torch
import torch.nn as nn

try:
    from dynamic_network_architectures.architectures.primus import (
        PrimusV2S,
        PrimusV2B,
        PrimusV2M,
        PrimusV2L,
    )
except ImportError:
    raise ImportError(
        "dynamic_network_architectures is required to run PrimusV2. "
        "Install via: pip install dynamic-network-architectures timm einops"
    )

def create_primus_v2(
    input_channels: int = 5,
    num_classes: int = 2,
    patch_size: Tuple[int, int, int] = (64, 128, 128),
    model_id: str = "S",
    patch_embed_size: Tuple[int, int, int] = (8, 8, 8),
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """
    Factory function to initialize a standardized PrimusV2 model.
    
    Args:
        input_channels: Number of input DCE-MRI phases (default: 5).
        num_classes: Number of segmentation classes (default: 2: background, tumor).
        patch_size: 3D spatial patch size for training (must be divisible by 8, e.g. [64, 128, 128]).
        model_id: Model capacity variant ('S': Small ~24.8M, 'B': Base ~75M, 'M': Medium ~147M).
        patch_embed_size: Patch token spatial downsampling size (default: 8x8x8).
        drop_path_rate: Stochastic depth rate (default: 0.1 for regularization).
    """
    model_id = model_id.upper()
    assert all(s % 8 == 0 for s in patch_size), f"All patch_size dimensions must be divisible by 8, got {patch_size}"

    if model_id == "S":
        model_cls = PrimusV2S
    elif model_id == "B":
        model_cls = PrimusV2B
    elif model_id == "M":
        model_cls = PrimusV2M
    elif model_id == "L":
        model_cls = PrimusV2L
    else:
        raise ValueError(f"Unknown model_id: {model_id}. Choose from ['S', 'B', 'M', 'L']")

    model = model_cls(
        input_channels=input_channels,
        output_channels=num_classes,
        patch_embed_size=patch_embed_size,
        input_shape=patch_size,
        drop_path_rate=drop_path_rate,
    )
    return model

if __name__ == "__main__":
    patch_size = (64, 128, 128)
    model = create_primus_v2(input_channels=5, num_classes=2, patch_size=patch_size, model_id="S")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"PrimusV2-S successfully initialized! Trainable parameters: {params:.2f}M")
    dummy = torch.randn(1, 5, *patch_size)
    out = model(dummy)
    print(f"Input shape: {dummy.shape} -> Output shape: {out.shape}")
    assert out.shape == (1, 2, *patch_size)
    print("Self-test passed!")
