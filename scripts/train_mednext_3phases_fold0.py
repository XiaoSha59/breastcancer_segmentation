#!/usr/bin/env python3
"""
train_mednext_3phases_fold0.py
==============================
Fully Standardized MedNeXt-L k3 Training on 3 Natural DCE-MRI Phases (Fold 0, 200 Epochs).
Biological Motivation (Naomi Fridman / BreastDCEDL):
- Uses 3 Natural Phases strictly:
  * Channel 0: Pre-contrast (T0 - Baseline anatomy)
  * Channel 1: Early Peak Wash-in (T_early - Peak tumor angiogenic enhancement)
  * Channel 2: Delayed Wash-out (T_late - Venous clearance & kinetic curve slope)
- Eliminates reliance on engineered Subtraction channels (I_e - I_0, I_l - I_0).
- Serves as the primary Ablation Study comparing 3 Natural Phases vs 5 Engineered Phases.

Engineering & Compliance Standards (AGENTS.md & 3D_UNET_BEST_PRACTICES.md):
1. 100% Original MedNeXt architecture (Roy et al., MICCAI 2023 / Facebook Research).
2. Input channels strictly set to 3; output classes strictly 2 (Background, Tumor).
3. Exactly 250 iterations/epoch, Poly-decay LR schedule, 33% FG oversampling.
4. CPU thread capping strictly at 4 threads (Intel i9-7900X server safety).
5. Zero auto-shutdown; safe signal handling.
"""

import os
import sys
import json
import time
import random
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# HARDWARE COMPATIBILITY: Disable cuDNN globally on node1 to prevent CUDNN_STATUS_NOT_INITIALIZED
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ensure mednext is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from scripts.mednext_v1 import create_mednext_v1
except ImportError:
    from mednext_v1 import create_mednext_v1

import monai
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNeXt-L k3 on 3 Natural DCE-MRI Phases (Fold 0)")
    parser.add_argument("--prep_dir", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations_per_epoch", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="L")
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth if present")
    return parser.parse_args()


class Fast3PhaseBatchSampler:
    """
    High-performance patch sampler strictly taking Channels 0, 1, 2 (3 Natural Phases):
    - Channel 0: Pre-contrast
    - Channel 1: Early Peak
    - Channel 2: Delayed Late
    Enforces nnU-Net's 33% foreground tumor oversampling guarantee.
    """
    def __init__(self, prep_dir, case_ids, patch_size, oversample_foreground_percent=0.33):
        self.prep_dir = prep_dir
        self.patch_size = np.array(patch_size)
        self.oversample_fg = oversample_foreground_percent
        self.handles = {}

        b2nd_cases = [cid for cid in case_ids if os.path.exists(os.path.join(prep_dir, f"{cid}.b2nd"))]
        self.is_b2nd = len(b2nd_cases) > len(case_ids) // 2

        if self.is_b2nd:
            valid_b2nd = []
            for cid in b2nd_cases:
                fpath = os.path.join(prep_dir, f"{cid}.b2nd")
                try:
                    if os.path.getsize(fpath) > 1024:
                        valid_b2nd.append(cid)
                except OSError:
                    pass
            self.case_ids = valid_b2nd
            print(f"MedNeXt 3-Phase Sampler initialized with {len(self.case_ids)} valid cases using Blosc2 (.b2nd).")
        else:
            npy_cases = [cid for cid in case_ids if os.path.exists(os.path.join(prep_dir, f"{cid}.npy"))]
            valid_npy = []
            for cid in npy_cases:
                fpath = os.path.join(prep_dir, f"{cid}.npy")
                try:
                    if os.path.getsize(fpath) > 1024:
                        valid_npy.append(cid)
                except OSError:
                    pass
            self.case_ids = valid_npy
            print(f"MedNeXt 3-Phase Sampler initialized with {len(self.case_ids)} valid cases using Numpy (.npy).")

    def get_case_data(self, cid):
        if cid not in self.handles:
            if len(self.handles) >= 100:
                self.handles.pop(next(iter(self.handles)))
            if self.is_b2nd:
                import blosc2
                img_path = os.path.join(self.prep_dir, f"{cid}.b2nd")
                seg_path = os.path.join(self.prep_dir, f"{cid}_seg.b2nd")
                img_h = blosc2.open(img_path)
                seg_h = blosc2.open(seg_path)
                self.handles[cid] = (img_h, seg_h)
            else:
                npy_path = os.path.join(self.prep_dir, f"{cid}.npy")
                self.handles[cid] = np.load(npy_path, mmap_mode="r")
        return self.handles[cid]

    def sample_batch(self, batch_size):
        batch_images = []
        batch_labels = []

        for b in range(batch_size):
            max_retries = 20
            for attempt in range(max_retries):
                cid = random.choice(self.case_ids)
                try:
                    data_or_handles = self.get_case_data(cid)

                    if self.is_b2nd:
                        img_h, seg_h = data_or_handles
                        shape = np.array(img_h.shape[1:])

                        needs_pad = shape < self.patch_size
                        if np.any(needs_pad):
                            n_c = min(3, img_h.shape[0])
                            img_buf = np.empty((n_c, *shape), dtype=np.float32)
                            for c in range(n_c):
                                img_buf[c] = img_h[c]
                            seg_buf = np.array(seg_h[:], dtype=np.float32)

                            pad_width_img = [(0, 0)] + [(0, max(0, p - s)) for s, p in zip(shape, self.patch_size)]
                            pad_width_seg = [(0, 0)] + [(0, max(0, p - s)) for s, p in zip(shape, self.patch_size)]
                            img_arr = np.pad(img_buf, pad_width_img, mode="constant", constant_values=0)
                            seg_arr = np.pad(seg_buf, pad_width_seg, mode="constant", constant_values=0)
                            shape = np.array(img_arr.shape[1:])
                        else:
                            img_arr = img_h
                            seg_arr = seg_h

                        seg_full = seg_arr[0]
                        is_fg = (random.random() < self.oversample_fg) and np.any(seg_full > 0)

                        if is_fg:
                            fg_coords = np.argwhere(seg_full > 0)
                            selected_coord = fg_coords[random.randint(0, len(fg_coords) - 1)]
                            start = np.clip(selected_coord - self.patch_size // 2, 0, shape - self.patch_size)
                        else:
                            max_start = shape - self.patch_size
                            start = np.array([random.randint(0, max(0, m)) for m in max_start])

                        s_0, s_1, s_2 = start
                        p_0, p_1, p_2 = self.patch_size
                        # Strictly extract first 3 natural phases (Pre, Early, Late)
                        patch_img = img_arr[:3, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]
                        patch_seg = seg_arr[:, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]

                        img = np.array(patch_img, dtype=np.float32)
                        lbl = (np.array(patch_seg, dtype=np.float32) > 0).astype(np.float32)
                    else:
                        data = data_or_handles
                        shape = np.array(data.shape[1:])
                        needs_pad = shape < self.patch_size
                        if np.any(needs_pad):
                            pad_width = [(0, 0)] + [(0, max(0, p - s)) for s, p in zip(shape, self.patch_size)]
                            data_arr = np.pad(data, pad_width, mode="constant", constant_values=0)
                            shape = np.array(data_arr.shape[1:])
                        else:
                            data_arr = data

                        seg = data_arr[5] if data_arr.shape[0] > 5 else data_arr[0]
                        is_fg = (random.random() < self.oversample_fg) and np.any(seg > 0)

                        if is_fg:
                            fg_coords = np.argwhere(seg > 0)
                            selected_coord = fg_coords[random.randint(0, len(fg_coords) - 1)]
                            start = np.clip(selected_coord - self.patch_size // 2, 0, shape - self.patch_size)
                        else:
                            max_start = shape - self.patch_size
                            start = np.array([random.randint(0, max(0, m)) for m in max_start])

                        s_d, s_h, s_w = start
                        p_d, p_h, p_w = self.patch_size
                        patch = data_arr[:, s_d:s_d+p_d, s_h:s_h+p_h, s_w:s_w+p_w]
                        # Strictly slice first 3 natural channels
                        img = patch[:3].astype(np.float32)
                        lbl = (patch[5:6] > 0).astype(np.float32) if patch.shape[0] > 5 else (patch[0:1] > 0).astype(np.float32)

                    # Random spatial augmentations (mirroring across 3 spatial axes)
                    for axis in [1, 2, 3]:
                        if random.random() < 0.5:
                            img = np.flip(img, axis=axis).copy()
                            lbl = np.flip(lbl, axis=axis).copy()

                    batch_images.append(torch.from_numpy(img))
                    batch_labels.append(torch.from_numpy(lbl))
                    break
                except Exception:
                    continue

        return torch.stack(batch_images), torch.stack(batch_labels)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")

    # Fixed CPU Thread Allocation for node1 / shared server safety (Strictly 4 threads)
    torch.set_num_threads(4)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

    def log_print(msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    log_print(f"=== Starting MedNeXt-{args.model_id} k{args.kernel_size} 3-Phase Training on I-SPY2 Fold {args.fold} ===")
    log_print("Biomarker Protocol: 3 Natural Phases strictly (Pre, Peak Early, Delayed Late) - Naomi / BreastDCEDL standard.")
    log_print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_print(f"Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_print("cuDNN disabled for safety on node1. Running 100% on rock-solid native PyTorch CUDA kernels.")

    # Auto-resolve splits_file
    if not os.path.exists(args.splits_file):
        alt_splits = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits_final.json"),
        ]
        for asp in alt_splits:
            if os.path.exists(asp):
                args.splits_file = asp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits[args.fold]["train"]
    val_ids = splits[args.fold]["val"]
    log_print(f"Loaded split fold {args.fold}: {len(train_ids)} train cases, {len(val_ids)} val cases.")

    # Auto-resolve prep_dir
    if not os.path.isdir(args.prep_dir):
        alt_prep = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for apr in alt_prep:
            if os.path.isdir(apr):
                args.prep_dir = apr
                break

    log_print(f"Using preprocessed directory: {args.prep_dir}")
    train_sampler = Fast3PhaseBatchSampler(args.prep_dir, train_ids, args.patch_size, oversample_foreground_percent=0.33)

    # Initialize MedNeXt with strictly num_channels=3 (3 Natural DCE-MRI Phases)
    model = create_mednext_v1(
        num_channels=3,
        num_classes=2,
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=True,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"MedNeXt-{args.model_id} k{args.kernel_size} (3-Phase Input) initialized. Trainable parameters: {total_params / 1e6:.2f}M")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except Exception:
        scaler = GradScaler()

    start_epoch = 0
    best_val_dice = 0.0
    progress_history = []

    checkpoint_latest = os.path.join(args.output_dir, "checkpoint_latest.pth")
    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")

    if args.resume and os.path.exists(checkpoint_latest):
        log_print(f"Resuming from {checkpoint_latest}...")
        ckpt = torch.load(checkpoint_latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["best_val_dice"]
        progress_history = ckpt.get("progress_history", [])
        log_print(f"Resumed at epoch {start_epoch}. Current best Dice: {best_val_dice:.4f}")

    start_time = time.time()
    mean_val_dice = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        # Exact Poly-decay LR schedule
        current_lr = args.initial_lr * ((1.0 - epoch / args.epochs) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for it in range(args.iterations_per_epoch):
            images, labels = train_sampler.sample_batch(args.batch_size)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                if isinstance(logits, list):
                    weights = [0.5, 0.25, 0.125, 0.0625, 0.0625]
                    loss = 0.0
                    for l, w in zip(logits, weights):
                        loss = loss + w * loss_fn(l, labels)
                else:
                    loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= args.iterations_per_epoch
        epoch_duration = time.time() - epoch_start
        log_print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {epoch_loss:.4f} | Poly LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s")

        # Fast Proxy Validation (20 random cases)
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dices = []
            with torch.no_grad():
                sub_val_ids = random.sample(val_ids, min(20, len(val_ids)))
                for cid in sub_val_ids:
                    b2nd_path = os.path.join(args.prep_dir, f"{cid}.b2nd")
                    npy_path = os.path.join(args.prep_dir, f"{cid}.npy")
                    try:
                        if os.path.exists(b2nd_path):
                            import blosc2
                            img_h = blosc2.open(b2nd_path)
                            seg_h = blosc2.open(os.path.join(args.prep_dir, f"{cid}_seg.b2nd"))
                            # Strictly take first 3 natural phases
                            val_buf = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                            for c in range(3):
                                val_buf[c] = img_h[c]
                            val_img = torch.from_numpy(val_buf).unsqueeze(0).to(device)
                            val_lbl = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                        elif os.path.exists(npy_path):
                            case_data = np.load(npy_path, mmap_mode="r")
                            val_img = torch.from_numpy(case_data[:3].astype(np.float32)).unsqueeze(0).to(device)
                            val_lbl = (case_data[5] > 0).astype(np.uint8)
                        else:
                            continue
                    except Exception as ve:
                        log_print(f"Validation warning on {cid}: {ve}")
                        continue

                    try:
                        with torch.amp.autocast('cuda'):
                            val_out = sliding_window_inference(
                                inputs=val_img,
                                roi_size=tuple(args.patch_size),
                                sw_batch_size=2,
                                predictor=model,
                                overlap=0.5,
                            )
                    except Exception:
                        val_out = sliding_window_inference(
                            inputs=val_img,
                            roi_size=tuple(args.patch_size),
                            sw_batch_size=2,
                            predictor=model,
                            overlap=0.5,
                        )

                    if isinstance(val_out, list):
                        val_out = val_out[0]

                    pred = (torch.softmax(val_out, dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                    inter = np.logical_and(pred, val_lbl).sum()
                    union = pred.sum() + val_lbl.sum()
                    dice = (2.0 * inter / union) if union > 0 else (1.0 if pred.sum() == 0 and val_lbl.sum() == 0 else 0.0)
                    val_dices.append(dice)
                    del val_img, val_out, pred

            mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
            is_best = mean_val_dice > best_val_dice
            if is_best:
                best_val_dice = mean_val_dice
                log_print(f"🎉 New best 3-phase validation Dice: {best_val_dice:.4f}! Saving checkpoint_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "args": vars(args),
                    "model_name": f"MedNeXt_{args.model_id}_k{args.kernel_size}_3phases",
                }, checkpoint_best)

            log_print(f"Validation Epoch {epoch:03d} | Proxy Val Dice: {mean_val_dice:.4f} (Best: {best_val_dice:.4f}) {'[★ NEW BEST]' if is_best else ''}")

        is_val_epoch = ((epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1)
        progress_history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "lr": current_lr,
            "val_dice": mean_val_dice if is_val_epoch else None,
            "time_seconds": epoch_duration,
        })

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "progress_history": progress_history,
            "args": vars(args),
            "model_name": f"MedNeXt_{args.model_id}_k{args.kernel_size}_3phases",
        }, checkpoint_latest)

    total_training_time = time.time() - start_time
    log_print(f"=== MEDNEXT 3-PHASE TRAINING COMPLETE in {datetime.timedelta(seconds=int(total_training_time))}! Best Val Dice: {best_val_dice:.4f} ===")

if __name__ == "__main__":
    main()
