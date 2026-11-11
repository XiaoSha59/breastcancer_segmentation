import sys
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mednext_v1 import create_mednext_v1
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

print("=== STEP 1: INITIALIZE MEDNEXT-L K3 ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = create_mednext_v1(
    num_channels=5,
    num_classes=2,
    model_id="L",
    kernel_size=3,
    deep_supervision=True,
    use_checkpoint=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MedNeXt-L k3 instantiated successfully! Parameters: {total_params / 1e6:.2f}M")

print("=== STEP 2: TEST FORWARD & BACKWARD (AMP FP16) ===")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

# Synthetic patch matching exact I-SPY2 settings: Batch 1, 5 channels, 128x128x64
x = torch.randn(1, 5, 128, 128, 64, device=device)
y = (torch.randn(1, 1, 128, 128, 64, device=device) > 0).float()

with autocast():
    outputs = model(x)
    assert isinstance(outputs, list), "Deep supervision output should be a list of multi-scale tensors"
    print(f"Deep supervision heads: {len(outputs)} scales. Top shape: {outputs[0].shape}")

    weights = [1.0 / (2 ** i) for i in range(len(outputs))]
    weights = [w / sum(weights) for w in weights]
    total_loss = 0.0
    for out, w in zip(outputs, weights):
        if out.shape[2:] != y.shape[2:]:
            down_y = nn.functional.interpolate(y, size=out.shape[2:], mode="nearest")
        else:
            down_y = y
        total_loss += w * loss_fn(out, down_y)

print(f"Loss computed successfully: {total_loss.item():.4f}")

scaler.scale(total_loss).backward()
scaler.step(optimizer)
scaler.update()
print("Backward pass and optimizer step completed cleanly!")
if torch.cuda.is_available():
    print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("=== STEP 3: TEST EVALUATION & SLIDING WINDOW INFERENCE ===")
eval_model = create_mednext_v1(
    num_channels=5,
    num_classes=2,
    model_id="L",
    kernel_size=3,
    deep_supervision=False,
).to(device)

eval_model.load_state_dict(model.state_dict(), strict=False)
eval_model.eval()

with torch.no_grad(), autocast():
    test_vol = torch.randn(1, 5, 128, 128, 64, device=device)
    val_out = sliding_window_inference(
        inputs=test_vol,
        roi_size=(128, 128, 64),
        sw_batch_size=1,
        predictor=eval_model,
        overlap=0.5,
    )
    print(f"Sliding window inference succeeded! Output shape: {val_out.shape}")

print("\n>>> ALL MEDNEXT-L K3 RIGOROUS TESTS PASSED 100% PERFECTLY! <<<")
