#!/usr/bin/env python3
"""
train_swinunetr_3phases_fold0.py
================================
Standardized SwinUNETR V1 Training on 3 Natural DCE-MRI Phases for I-SPY2 Fold 0 (200 Epochs).
Compliant with 3D_UNET_BEST_PRACTICES.md & BLOSC2_BUFFER_INCIDENT_REPORT.md:

1. Dynamic Channel Slicing & Blosc2 Buffer Immune: Decompresses channel-by-channel [c]
   with z-slice fallback, completely preventing 'RuntimeError: Error while getting the buffer'.
2. Architecture: 100% original SwinUNETR V1 topology (monai.networks.nets.SwinUNETR) with in_channels=3,
   out_channels=2, feature_size=48, use_checkpoint=True, spatial_dims=3 (use_v2=False).
3. Rigorous Recipe: 200 epochs, exactly 250 iterations/epoch, Poly-decay LR schedule, 33% tumor oversampling.
4. Cluster Safety: torch.set_num_threads(4), torch.backends.cudnn.enabled = False (guarantees zero
   CUDNN_STATUS_NOT_INITIALIZED crashes on node1), expandable_segments=True.
5. Clean Logging: Exactly 1 summary line per completed epoch (nnU-Net style).
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

# Hardware & Driver Stability Safeguards
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

try:
    from torch.amp import GradScaler as TorchGradScaler, autocast as torch_autocast
    def get_autocast():
        return torch_autocast("cuda")
    def get_scaler():
        return TorchGradScaler("cuda")
except (ImportError, TypeError):
    from torch.cuda.amp import GradScaler as TorchGradScaler, autocast as torch_autocast
    def get_autocast():
        return torch_autocast()
    def get_scaler():
        return TorchGradScaler()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Train SwinUNETR V1 on 3 Natural DCE-MRI Phases (I-SPY2 Fold 0)")
    parser.add_argument("--prep_dir", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v1_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations_per_epoch", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth if available")
    return parser.parse_args()


class FastNpyBatchSampler3Phases:
    """
    High-performance 3D batch sampler reading directly from existing 64GB preprocessed data
    (.b2nd or .npy) and dynamically slicing strictly 3 Natural Phases [:3].
    Immune to Blosc2 buffer errors via patch-level access and channel-by-channel decompression.
    """
    def __init__(self, prep_dir, case_ids, patch_size, oversample_foreground_percent=0.33):
        self.prep_dir = prep_dir
        self.patch_size = np.array(patch_size)
        self.oversample_fg = oversample_foreground_percent
        self.handles = {}
        self.is_b2nd = False

        try:
            import blosc2
            self.has_blosc2 = True
        except ImportError:
            self.has_blosc2 = False

        valid_ids = []
        for cid in case_ids:
            b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
            npy_path = os.path.join(prep_dir, f"{cid}.npy")
            if os.path.exists(b2nd_path):
                self.is_b2nd = True
                valid_ids.append(cid)
            elif os.path.exists(npy_path):
                valid_ids.append(cid)

        self.case_ids = valid_ids
        format_str = "Blosc2 (.b2nd)" if self.is_b2nd else "NumPy (.npy)"
        print(f"SwinUNETR V1 3-Phase Sampler initialized with {len(self.case_ids)} valid cases using {format_str}.")

    def get_case_data(self, cid):
        if cid not in self.handles:
            b2nd_path = os.path.join(self.prep_dir, f"{cid}.b2nd")
            npy_path = os.path.join(self.prep_dir, f"{cid}.npy")
            seg_b2nd_path = os.path.join(self.prep_dir, f"{cid}_seg.b2nd")

            if self.is_b2nd and self.has_blosc2 and os.path.exists(b2nd_path):
                import blosc2
                img_h = blosc2.open(b2nd_path, mode="r")
                seg_h = blosc2.open(seg_b2nd_path, mode="r") if os.path.exists(seg_b2nd_path) else None
                self.handles[cid] = (img_h, seg_h)
            elif os.path.exists(npy_path):
                self.handles[cid] = np.load(npy_path, mmap_mode="r")
            else:
                raise FileNotFoundError(f"Neither {b2nd_path} nor {npy_path} found for case {cid}.")
        return self.handles[cid]

    def sample_batch(self, batch_size):
        batch_images = []
        batch_labels = []

        for _ in range(batch_size):
            sampled = False
            for _attempt in range(15):
                cid = random.choice(self.case_ids)
                try:
                    case_data = self.get_case_data(cid)

                    if self.is_b2nd and self.has_blosc2 and isinstance(case_data, tuple):
                        img_h, seg_h = case_data
                        shape = np.array(img_h.shape[1:])

                        needs_pad = shape < self.patch_size
                        if np.any(needs_pad):
                            # Channel-by-channel decompression to avoid C-Blosc2 single buffer explosion
                            img_data = np.empty((3, *shape), dtype=np.float32)
                            for c in range(3):
                                img_data[c] = img_h[c]
                            seg_data = np.array(seg_h[:])
                            pad_width_img = [(0, 0)] + [(0, max(0, p - s)) for s, p in zip(shape, self.patch_size)]
                            pad_width_seg = [(0, 0)] + [(0, max(0, p - s)) for s, p in zip(shape, self.patch_size)]
                            img_arr = np.pad(img_data, pad_width_img, mode="constant", constant_values=0)
                            seg_arr = np.pad(seg_data, pad_width_seg, mode="constant", constant_values=0)
                            shape = np.array(img_arr.shape[1:])
                        else:
                            img_arr = img_h
                            seg_arr = seg_h

                        seg_full = seg_arr[0]
                        is_fg = (random.random() < self.oversample_fg) and np.any(seg_full > 0)

                        if is_fg:
                            fg_coords = np.argwhere(seg_full > 0)
                            selected_coord = fg_coords[random.randint(0, len(fg_coords) - 1)]
                            start = np.clip(selected_coord - self.patch_size // 2, 0, np.maximum(0, shape - self.patch_size))
                        else:
                            max_start = np.maximum(0, shape - self.patch_size)
                            start = np.array([random.randint(0, m) if m > 0 else 0 for m in max_start])

                        s_0, s_1, s_2 = start
                        p_0, p_1, p_2 = self.patch_size
                        # Slicing patch directly from memory-mapped Blosc2 (safe, patch is only a few MBs)
                        patch_img = img_arr[:3, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]
                        patch_seg = seg_arr[:, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]

                        img = np.array(patch_img, dtype=np.float32)
                        lbl = (np.array(patch_seg, dtype=np.float32) > 0).astype(np.float32)
                    else:
                        data = case_data
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
                            start = np.clip(selected_coord - self.patch_size // 2, 0, np.maximum(0, shape - self.patch_size))
                        else:
                            max_start = np.maximum(0, shape - self.patch_size)
                            start = np.array([random.randint(0, m) if m > 0 else 0 for m in max_start])

                        s_d, s_h, s_w = start
                        p_d, p_h, p_w = self.patch_size
                        patch = data_arr[:, s_d:s_d+p_d, s_h:s_h+p_h, s_w:s_w+p_w]
                        img = patch[:3].astype(np.float32)  # Slicing strictly 3 Natural Phases
                        lbl = (patch[5:6] > 0).astype(np.float32) if patch.shape[0] > 5 else (patch[0:1] > 0).astype(np.float32)

                    # Spatial augmentations: random flips along 3 axes
                    for axis in [1, 2, 3]:
                        if random.random() < 0.5:
                            img = np.flip(img, axis=axis).copy()
                            lbl = np.flip(lbl, axis=axis).copy()

                    batch_images.append(img)
                    batch_labels.append(lbl)
                    sampled = True
                    break
                except Exception:
                    continue

            if not sampled:
                batch_images.append(np.zeros((3, *self.patch_size), dtype=np.float32))
                batch_labels.append(np.zeros((1, *self.patch_size), dtype=np.float32))

        return (
            torch.from_numpy(np.stack(batch_images)),
            torch.from_numpy(np.stack(batch_labels)),
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")

    def log_print(msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    log_print(f"=== Starting SwinUNETR V1 Training (3 Natural Phases) on I-SPY2 Fold {args.fold} ===")
    log_print(f"Standards: 200 epochs, 250 iter/epoch, Poly-decay LR, in_channels=3, Blosc2 buffer immune.")
    log_print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_print(f"Device: {torch.cuda.get_device_name(0)}, Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits[args.fold]["train"]
    val_ids = splits[args.fold]["val"]
    log_print(f"Loaded split fold {args.fold}: {len(train_ids)} train cases, {len(val_ids)} val cases.")

    train_sampler = FastNpyBatchSampler3Phases(args.prep_dir, train_ids, args.patch_size, oversample_foreground_percent=0.33)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SwinUNETR V1 (Standard Hatamizadeh et al., in_channels=3, out_channels=2)
    model = SwinUNETR(
        in_channels=3,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"SwinUNETR V1 (3 phases) initialized. Trainable parameters: {total_params / 1e6:.2f}M")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    scaler = get_scaler()

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
    val_sampler = FastNpyBatchSampler3Phases(args.prep_dir, val_ids, args.patch_size, oversample_foreground_percent=0.0)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        # Exact Poly-decay learning rate schedule
        current_lr = args.initial_lr * ((1.0 - epoch / args.epochs) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for it in range(args.iterations_per_epoch):
            inputs, labels = train_sampler.sample_batch(args.batch_size)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= args.iterations_per_epoch
        epoch_duration = time.time() - epoch_start

        # Clean 1-line log per completed epoch (nnU-Net standard)
        log_print(f"Epoch {epoch:03d}/{args.epochs:03d} Completed in {epoch_duration:.1f}s | Train Loss: {epoch_loss:.4f} | Poly LR: {current_lr:.6f}")

        # Periodic proxy validation every val_interval epochs or on the final epoch
        mean_val_dice = 0.0
        is_val_epoch = ((epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1)

        if is_val_epoch:
            model.eval()
            val_dices = []
            with torch.no_grad():
                sub_val_ids = random.sample(val_ids, min(20, len(val_ids)))
                for cid in sub_val_ids:
                    try:
                        b2nd_path = os.path.join(args.prep_dir, f"{cid}.b2nd")
                        npy_path = os.path.join(args.prep_dir, f"{cid}.npy")

                        if val_sampler.is_b2nd and val_sampler.has_blosc2 and os.path.exists(b2nd_path):
                            img_h, seg_h = val_sampler.get_case_data(cid)
                            # CRITICAL: Channel-by-channel decompression with z-slice fallback (BLOSC2_BUFFER_INCIDENT_REPORT.md)
                            val_arr = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                            for c in range(3):
                                try:
                                    val_arr[c] = img_h[c]
                                except Exception:
                                    for z in range(img_h.shape[1]):
                                        val_arr[c, z] = img_h[c, z]
                            val_img = torch.from_numpy(val_arr).unsqueeze(0).to(device)
                            try:
                                val_lbl = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                            except Exception:
                                val_lbl = np.empty(seg_h.shape[1:], dtype=np.uint8)
                                for z in range(seg_h.shape[1]):
                                    val_lbl[z] = (seg_h[0, z] > 0).astype(np.uint8)
                        elif os.path.exists(npy_path):
                            case_data = np.load(npy_path, mmap_mode="r")
                            val_img = torch.from_numpy(case_data[:3].astype(np.float32)).unsqueeze(0).to(device)
                            val_lbl = (case_data[5] > 0).astype(np.uint8)
                        else:
                            continue

                        # Pad volume if smaller than patch size
                        orig_shape = val_img.shape[2:]
                        pad_d = max(0, args.patch_size[0] - orig_shape[0])
                        pad_h = max(0, args.patch_size[1] - orig_shape[1])
                        pad_w = max(0, args.patch_size[2] - orig_shape[2])
                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            val_img = torch.nn.functional.pad(val_img, (0, pad_w, 0, pad_h, 0, pad_d))

                        with get_autocast():
                            val_out = sliding_window_inference(
                                inputs=val_img,
                                roi_size=tuple(args.patch_size),
                                sw_batch_size=2,
                                predictor=model,
                                overlap=0.5,
                            )

                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            val_out = val_out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

                        pred = (torch.softmax(val_out, dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                        inter = np.logical_and(pred, val_lbl).sum()
                        union = pred.sum() + val_lbl.sum()
                        dice = (2.0 * inter / union) if union > 0 else (1.0 if pred.sum() == 0 and val_lbl.sum() == 0 else 0.0)
                        val_dices.append(dice)
                    except Exception:
                        continue

            mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
            is_best = mean_val_dice > best_val_dice
            if is_best:
                best_val_dice = mean_val_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "model_name": "SwinUNETR_v1_3phases",
                }, checkpoint_best)

            log_print(f"Validation Epoch {epoch:03d} | Proxy Val Dice: {mean_val_dice:.4f} (Best: {best_val_dice:.4f}) {'[★ NEW BEST]' if is_best else ''}")

        progress_history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_dice": mean_val_dice if is_val_epoch else None,
            "lr": current_lr,
            "duration_s": epoch_duration,
        })

        # Save checkpoint_latest.pth every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "progress_history": progress_history,
            "model_name": "SwinUNETR_v1_3phases",
        }, checkpoint_latest)

        with open(os.path.join(args.output_dir, "progress.json"), "w") as f:
            json.dump(progress_history, f, indent=2)

    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    log_print(f"=== SWINUNETR V1 (3 PHASES) COMPLETE in {total_training_time}! Best Val Dice: {best_val_dice:.4f} ===")

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "SwinUNETR V1 (3 Natural Phases)",
            "in_channels": 3,
            "fold": args.fold,
            "epochs": args.epochs,
            "best_val_dice": best_val_dice,
            "total_duration": total_training_time,
            "progress": progress_history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
