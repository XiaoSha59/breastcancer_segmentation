#!/usr/bin/env python3
"""
debug_cudnn.py
Diagnose why cuDNN fails to initialize on PyTorch 2.6.0 + CUDA 12.4 + RTX 4090.
Tests:
1. Warmup with Conv2D
2. Different cuDNN settings: deterministic, benchmark, allow_tf32
3. cuDNN v8 vs v9 API flags
4. Direct F.conv3d calls with explicit float16 (half) vs float32
"""

import os
import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version:    {torch.version.cuda}")
print(f"cuDNN version:   {torch.backends.cudnn.version()}")
print(f"cuDNN enabled:   {torch.backends.cudnn.enabled}")

# Test 1: Conv2D with cuDNN
print("\n--- TEST 1: Conv2D with cuDNN ---")
try:
    torch.backends.cudnn.enabled = True
    c2 = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    x2 = torch.randn(2, 3, 32, 32, device='cuda')
    y2 = c2(x2)
    torch.cuda.synchronize()
    print("Conv2D cuDNN: SUCCESS!")
except Exception as e:
    print(f"Conv2D cuDNN FAILED: {type(e).__name__}: {e}")

# Test 2: Conv3D with FP16 (Half) in Autocast (Standard nnU-Net uses FP16 AMP)
print("\n--- TEST 2: Conv3D in FP16 (autocast) with cuDNN ---")
try:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    c3 = torch.nn.Conv3d(3, 32, 3, padding=1).cuda()
    x3 = torch.randn(2, 3, 20, 32, 32, device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        y3 = c3(x3)
    torch.cuda.synchronize()
    print("Conv3D FP16 cuDNN: SUCCESS!")
except Exception as e:
    print(f"Conv3D FP16 cuDNN FAILED: {type(e).__name__}: {e}")

# Test 3: Conv3D with FP32 and benchmark=False, deterministic=True
print("\n--- TEST 3: Conv3D FP32 benchmark=False, deterministic=True ---")
try:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    c3 = torch.nn.Conv3d(3, 32, 3, padding=1).cuda()
    x3 = torch.randn(2, 3, 20, 32, 32, device='cuda')
    y3 = c3(x3)
    torch.cuda.synchronize()
    print("Conv3D FP32 (deterministic) cuDNN: SUCCESS!")
except Exception as e:
    print(f"Conv3D FP32 (deterministic) cuDNN FAILED: {type(e).__name__}: {e}")

# Test 4: Check if cuDNN SDPA or V8 API is causing issues
print("\n--- TEST 4: Environment flags check ---")
print("TORCH_CUDNN_V8_API_ENABLED =", os.environ.get("TORCH_CUDNN_V8_API_ENABLED"))
print("nnUNet_compile =", os.environ.get("nnUNet_compile"))
print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

# Test 5: Can we use patch_size [192, 192, 64] with native CUDA (like standard nnU-Net)?
print("\n--- TEST 5: Native CUDA (cuDNN disabled) with patch [192, 192, 64] batch_size 2 ---")
torch.backends.cudnn.enabled = False
try:
    # Test decoder convolution at patch [192, 192, 64]
    c_dec = torch.nn.Conv3d(64, 32, 3, padding=1).cuda()
    x_dec = torch.randn(2, 64, 64, 96, 96, device='cuda')
    y_dec = c_dec(x_dec)
    torch.cuda.synchronize()
    print("Decoder conv with patch [192, 192, 64]: SUCCESS! (VRAM safe)")
except Exception as e:
    print(f"Decoder conv with patch [192, 192, 64] FAILED: {e}")
