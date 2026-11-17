#!/usr/bin/env python3
"""
train_swinunetr_v2_3phases_fold0.py
===================================
Standardized SwinUNETR V2 Training on 3 Natural DCE-MRI Phases for I-SPY2 Fold 0 (200 Epochs).
Compliant with 3D_UNET_BEST_PRACTICES.md Section 9.5:
1. Dynamic channel slicing: extracts [:3] (T0 Pre-contrast, T1 Peak Early, T2 Delayed Late) directly
   from existing nnUNetPlans_3d_fullres data without recreating 64GB dataset.
2. Architecture: monai.networks.nets.SwinUNETR with in_channels=3, out_channels=2, use_v2=True,
   downsample="mergingv2", feature_size=48.
3. 250 iterations/epoch, 33% tumor foreground oversampling, Poly-decay LR schedule (initial 1e-4).
4. torch.backends.cudnn.enabled = False (guarantees zero CUDNN_STATUS_NOT_INITIALIZED crashes).
5. CPU threads strictly limited: torch.set_num_threads(4) for shared cluster safety.
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

import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Train SwinUNETR V2 on 3 Natural Phases (I-SPY2 Fold 0)")
    parser.add_argument("--prep_dir", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v2_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations_per_epoch", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--use_checkpoint", action="store_true", default=False, help="Enable activation checkpointing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth if available")
    return parser.parse_args()

class FastNpyBatchSampler3Phases:
    """
    High-performance patch sampler directly reading memory-mapped .b2nd or .npy files
    and dynamically slicing strictly 3 Natural Phases [:3].
    """
    def __init__(self, prep_dir, case_ids, patch_size, oversample_foreground_percent=0.33):
        self.prep_dir = prep_dir
        self.case_ids = case_ids
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
        print(f"3-Phase Sampler initialized with {len(self.case_ids)} valid cases using {format_str}.")

    def get_case_data(self, cid):
        if cid not in self.handles:
            b2nd_path = os.path.join(self.prep_dir, f"{cid}.b2nd")
            npy_path = os.path.join(self.prep_dir, f"{cid}.npy")
            seg_b2nd_path = os.path.join(self.prep_dir, f"{cid}_seg.b2nd")

            if self.is_b2nd and self.has_blosc2 and os.path.exists(b2nd_path):
                import blosc2
                img_array = blosc2.open(b2nd_path, mode="r")
                seg_array = blosc2.open(seg_b2nd_path, mode="r") if os.path.exists(seg_b2nd_path) else None
                self.handles[cid] = (img_array, seg_array)
            elif os.path.exists(npy_path):
                data = np.load(npy_path, mmap_mode="r")
                self.handles[cid] = data
            else:
                raise FileNotFoundError(f"Neither {b2nd_path} nor {npy_path} found.")
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
                            # Slice strictly 3 channels [:3] channel-by-channel to avoid Blosc2 buffer limit
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
                        # Strictly 3 Natural Channels
                        patch_img = img_arr[:3, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]
                        patch_seg = seg_arr[:, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]

                        img_patch = np.array(patch_img, dtype=np.float32)
                        seg_patch = (np.array(patch_seg) > 0).astype(np.int64)
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
                        # Strictly slice [:3]
                        img_patch = patch[:3].astype(np.float32)
                        seg_patch = (patch[5:6] > 0).astype(np.int64)

                    # Spatial flips
                    for axis in [1, 2, 3]:
                        if random.random() < 0.5:
                            img_patch = np.flip(img_patch, axis=axis).copy()
                            seg_patch = np.flip(seg_patch, axis=axis).copy()

                    batch_images.append(torch.from_numpy(img_patch.astype(np.float32)))
                    batch_labels.append(torch.from_numpy(seg_patch.astype(np.int64)))
                    sampled = True
                    break
                except Exception as ex:
                    print(f"[DATA WARNING] Case {cid} failed during sampling ({ex}). Retrying...")
                    if cid in self.case_ids:
                        self.case_ids.remove(cid)
                    if cid in self.handles:
                        self.handles.pop(cid, None)

            if not sampled:
                raise RuntimeError(f"Failed to sample valid 3-phase patch after 15 attempts.")

        images_tensor = torch.stack(batch_images, dim=0)
        labels_tensor = torch.stack(batch_labels, dim=0)
        return images_tensor, labels_tensor

def get_poly_lr(initial_lr, current_epoch, max_epochs, exponent=0.9):
    return initial_lr * ((1.0 - float(current_epoch) / float(max_epochs)) ** exponent)

def main():
    args = parse_args()

    # Auto-resolve paths on node1 or GCP
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.prep_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        cand_preps = [
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for p in cand_preps:
            if os.path.exists(p):
                args.prep_dir = p
                break

    if args.splits_file.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        cand_splits = [
            os.path.join(os.path.dirname(args.prep_dir), "splits_final.json"),
            os.path.join(project_root, "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/splits_final.json"),
        ]
        for s in cand_splits:
            if os.path.exists(s):
                args.splits_file = s
                break

    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", "SwinUNETR_v2_3phases_200epochs", f"fold_{args.fold}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")

    # Cluster Safety: Fixed CPU Thread Allocation (strictly 4 threads)
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

    log_print(f"=== Starting SwinUNETR V2 (3 NATURAL PHASES) Training on I-SPY2 Fold {args.fold} ===")
    log_print(f"Arguments: {vars(args)}")

    if not os.path.exists(args.splits_file):
        raise FileNotFoundError(f"Splits file not found: {args.splits_file}")

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits[args.fold]["train"]
    val_ids = splits[args.fold]["val"]
    train_sampler = FastNpyBatchSampler3Phases(args.prep_dir, train_ids, args.patch_size, oversample_foreground_percent=0.33)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardware safety on RTX 4090 / L4: strictly disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = False
    log_print("Hardware safety: cuDNN disabled (torch.backends.cudnn.enabled = False). Running 100% on native PyTorch CUDA kernels.")

    # SwinUNETR V2: in_channels=3 (3 Natural Phases), out_channels=2, use_v2=True, downsample='mergingv2'
    log_print(f"Initializing SwinUNETR V2 for 3 Natural Phases (in_channels=3, out_channels=2, use_v2=True, downsample='mergingv2', feature_size=48)...")
    model = SwinUNETR(
        in_channels=3,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=3,
        downsample="mergingv2",
        use_v2=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"SwinUNETR V2 (3 Natural Phases) initialized. Trainable parameters: {total_params / 1e6:.2f}M")

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
        log_print(f"Resumed at epoch {start_epoch}. Current best Val Dice: {best_val_dice:.4f}")

    start_time = time.time()
    val_sampler = FastNpyBatchSampler3Phases(args.prep_dir, val_ids, args.patch_size, oversample_foreground_percent=0.0)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        current_lr = get_poly_lr(args.initial_lr, epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        running_loss = 0.0

        for it in range(args.iterations_per_epoch):
            images, labels = train_sampler.sample_batch(args.batch_size)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast():
                preds = model(images)
                loss = loss_fn(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / args.iterations_per_epoch
        epoch_duration = time.time() - epoch_start
        log_print(f"Epoch {epoch:03d}/{args.epochs:03d} Completed in {epoch_duration:.1f}s | Train Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")

        # Validation every val_interval epochs or last epoch
        mean_val_dice = 0.0
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dices = []
            log_print(f"Running validation on {len(val_ids)} cases (sliding window)...")

            with torch.no_grad():
                for cid in val_ids:
                    try:
                        b2nd_path = os.path.join(args.prep_dir, f"{cid}.b2nd")
                        npy_path = os.path.join(args.prep_dir, f"{cid}.npy")

                        if val_sampler.is_b2nd and val_sampler.has_blosc2 and os.path.exists(b2nd_path):
                            img_h, seg_h = val_sampler.get_case_data(cid)
                            val_arr = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                            for c in range(3):
                                val_arr[c] = img_h[c]
                            val_img = torch.from_numpy(val_arr).unsqueeze(0).to(device)
                            val_lbl = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                        elif os.path.exists(npy_path):
                            case_data = np.load(npy_path, mmap_mode="r")
                            val_img = torch.from_numpy(case_data[:3].astype(np.float32)).unsqueeze(0).to(device)
                            val_lbl = (case_data[5] > 0).astype(np.uint8)
                        else:
                            continue

                        with get_autocast():
                            val_out = sliding_window_inference(
                                inputs=val_img,
                                roi_size=tuple(args.patch_size),
                                sw_batch_size=2,
                                predictor=model,
                                overlap=0.5,
                            )

                        pred = (torch.softmax(val_out, dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                        inter = np.logical_and(pred, val_lbl).sum()
                        union = pred.sum() + val_lbl.sum()
                        dice = (2.0 * inter / union) if union > 0 else (1.0 if pred.sum() == 0 and val_lbl.sum() == 0 else 0.0)
                        val_dices.append(dice)
                    except Exception as e:
                        log_print(f"Warning: Failed to evaluate val case {cid}: {e}. Skipping.")
                        continue

            mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
            is_best = mean_val_dice > best_val_dice
            if is_best:
                best_val_dice = mean_val_dice
                log_print(f"🎉 New best validation Dice achieved: {best_val_dice:.4f}! Saving checkpoint_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "args": vars(args),
                    "model_name": "SwinUNETR_v2_3phases",
                }, checkpoint_best)

            log_print(f"Validation Epoch {epoch:03d} | Proxy Val Dice: {mean_val_dice:.4f} (Best: {best_val_dice:.4f}) {'[★ NEW BEST]' if is_best else ''}")

        # Save checkpoint_latest.pth every epoch
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
            "model_name": "SwinUNETR_v2_3phases",
        }, checkpoint_latest)

        with open(os.path.join(args.output_dir, "progress.json"), "w") as f:
            json.dump(progress_history, f, indent=2)

    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    log_print(f"=== SWINUNETR V2 (3 PHASES) TRAINING COMPLETE in {total_training_time}! Best Val Dice: {best_val_dice:.4f} ===")

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": "SwinUNETR (V2, 3 Natural Phases)",
            "use_v2": True,
            "downsample": "mergingv2",
            "in_channels": 3,
            "epochs": args.epochs,
            "best_val_dice": best_val_dice,
            "training_time": total_training_time,
            "completed_at": datetime.datetime.now().isoformat(),
        }, f, indent=2)

if __name__ == "__main__":
    main()
