#!/usr/bin/env python3
"""
train_primus_v2_fold0.py
========================
Bulletproof Standardized PrimusV2-S Training on I-SPY2 Fold 0 (200 Epochs).
Compliant with 3D_UNET_BEST_PRACTICES.md:
1. 100% REUSE of 64GB preprocessed .b2nd / .npy data in nnUNetPlans_3d_fullres.
2. Exactly 250 iterations per epoch, Poly-decay LR schedule.
3. 33% foreground oversampling per batch.
4. Safe int64 labels & dimension-safe DiceCELoss (zero dtype/shape mismatch).
5. Gradient clipping (max_norm=12.0) for stable Transformer optimization.
6. Exception-shielded validation with dynamic padding, unpadding, and cleanup.
7. In-place checkpoint overwrites (~250 MB total disk usage).
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
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from primus_v2 import create_primus_v2

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Train PrimusV2-S on I-SPY2 Fold 0")
    parser.add_argument("--prep_dir", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations_per_epoch", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--model_id", type=str, default="S", choices=["S", "B", "M"])
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pth if available")
    return parser.parse_args()

class RobustDiceCELoss(nn.Module):
    """
    Rock-solid composite loss combining DiceLoss and CrossEntropyLoss.
    Guarantees zero crashes from dimension mismatch (B, 1, ...) vs (B, 2, ...).
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, 2, D, H, W)
        # target: (B, 1, D, H, W) or (B, D, H, W) of type int64
        dice_loss = self.dice(pred, target)
        if target.dim() == 5 and target.shape[1] == 1:
            target_ce = target[:, 0].long()
        else:
            target_ce = target.long()
        ce_loss = self.ce(pred, target_ce)
        return dice_loss + ce_loss

class FastNpyBatchSampler:
    """
    High-performance patch sampler directly reading memory-mapped .b2nd or .npy files.
    Enforces nnU-Net's golden rule: 33% of patches guaranteed to contain tumor foreground.
    """
    def __init__(self, prep_dir, case_ids, patch_size, oversample_foreground_percent=0.33):
        self.prep_dir = prep_dir
        self.patch_size = np.array(patch_size)
        self.oversample_fg = oversample_foreground_percent
        self.handles = {}
        self.is_b2nd = False

        b2nd_cases = [cid for cid in case_ids if os.path.exists(os.path.join(prep_dir, f"{cid}.b2nd"))]
        if len(b2nd_cases) > 0:
            self.is_b2nd = True
            valid_b2nd = []
            for cid in b2nd_cases:
                fpath = os.path.join(prep_dir, f"{cid}.b2nd")
                spath = os.path.join(prep_dir, f"{cid}_seg.b2nd")
                try:
                    if os.path.getsize(fpath) > 1024 and os.path.getsize(spath) > 1024:
                        valid_b2nd.append(cid)
                except OSError:
                    pass
            self.case_ids = valid_b2nd
            print(f"PrimusV2 Sampler initialized with {len(self.case_ids)} valid cases using Blosc2 (.b2nd).")
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
            print(f"PrimusV2 Sampler initialized with {len(self.case_ids)} valid cases using Numpy (.npy).")

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

        pz, py, px = self.patch_size

        for b_idx in range(batch_size):
            max_retries = 20
            sampled = False
            for attempt in range(max_retries):
                cid = random.choice(self.case_ids)
                try:
                    case_data = self.get_case_data(cid)

                    if self.is_b2nd:
                        img_h, seg_h = case_data
                        shape = np.array(img_h.shape[1:])

                        needs_pad = shape < self.patch_size
                        if np.any(needs_pad):
                            img_data = np.array(img_h[:])
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
                        patch_img = img_arr[:, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]
                        patch_seg = seg_arr[:, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]

                        img_patch = np.array(patch_img, dtype=np.float32)
                        seg_patch = np.array(patch_seg, dtype=np.int64)
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
                        img_patch = patch[:5].astype(np.float32)
                        seg_patch = patch[5:6].astype(np.int64)

                    # Labels strictly int64 (0 or 1)
                    lbl_arr = (seg_patch > 0).astype(np.int64)
                    img_arr = img_patch.astype(np.float32)

                    # Spatial flips
                    for axis in [1, 2, 3]:
                        if random.random() < 0.5:
                            img_arr = np.flip(img_arr, axis=axis).copy()
                            lbl_arr = np.flip(lbl_arr, axis=axis).copy()

                    batch_images.append(img_arr)
                    batch_labels.append(lbl_arr)
                    sampled = True
                    break
                except Exception as ex:
                    if cid in self.case_ids:
                        self.case_ids.remove(cid)
                    if cid in self.handles:
                        self.handles.pop(cid, None)

            if not sampled:
                raise RuntimeError(f"Failed to sample a valid batch after {max_retries} attempts.")

        return (
            torch.from_numpy(np.stack(batch_images)),
            torch.from_numpy(np.stack(batch_labels)),
        )

def main():
    args = parse_args()

    # Auto-resolve splits_file if default does not exist
    if not os.path.exists(args.splits_file):
        candidate_splits = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for csp in candidate_splits:
            if os.path.exists(csp):
                args.splits_file = csp
                break

    # Auto-resolve prep_dir if default does not exist
    if not os.path.isdir(args.prep_dir):
        candidate_prep = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for cpr in candidate_prep:
            if os.path.isdir(cpr):
                args.prep_dir = cpr
                break

    # Auto-resolve output_dir if default is /home/XiaoSha/... on another machine
    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", f"PrimusV2_{args.model_id}_200epochs", f"fold_{args.fold}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")

    # Fixed CPU Thread Allocation for node1 (Intel i9-7900X): strictly 4 threads
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

    log_print(f"=== Starting PrimusV2-{args.model_id} Training on I-SPY2 Fold {args.fold} ===")
    log_print(f"Architecture: Pure 3D Transformer with Iterative Tokenizer & 3D Axial RoPE (DKFZ 2024).")
    log_print(f"Data Recipe: 250 iter/epoch, Poly LR, 5 channels, 64GB .b2nd/.npy preprocessed data.")
    log_print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_print(f"Device: {torch.cuda.get_device_name(0)}, Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if not os.path.exists(args.splits_file):
        raise FileNotFoundError(f"Splits file not found: {args.splits_file}")

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits[args.fold]["train"]
    val_ids = splits[args.fold]["val"]
    log_print(f"Loaded split fold {args.fold}: {len(train_ids)} train cases, {len(val_ids)} val cases.")

    train_sampler = FastNpyBatchSampler(args.prep_dir, train_ids, args.patch_size, oversample_foreground_percent=0.33)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = create_primus_v2(
        input_channels=5,
        num_classes=2,
        patch_size=tuple(args.patch_size),
        model_id=args.model_id,
        drop_path_rate=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"PrimusV2-{args.model_id} initialized. Trainable parameters: {total_params / 1e6:.2f}M")

    loss_fn = RobustDiceCELoss()
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

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        current_lr = args.initial_lr * ((1.0 - epoch / args.epochs) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for it in range(args.iterations_per_epoch):
            inputs, labels = train_sampler.sample_batch(args.batch_size)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= args.iterations_per_epoch
        epoch_duration = time.time() - epoch_start

        log_print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {epoch_loss:.4f} | Poly LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s")

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
                            val_img = torch.from_numpy(np.array(img_h[:], dtype=np.float32)).unsqueeze(0).to(device)
                            val_lbl = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                        elif os.path.exists(npy_path):
                            case_data = np.load(npy_path, mmap_mode="r")
                            val_img = torch.from_numpy(case_data[:5].astype(np.float32)).unsqueeze(0).to(device)
                            val_lbl = (case_data[5] > 0).astype(np.uint8)
                        else:
                            continue

                        # Dynamic padding
                        pz, py, px = args.patch_size
                        _, orig_c, orig_d, orig_h, orig_w = val_img.shape
                        pad_d = max(0, pz - orig_d)
                        pad_h = max(0, py - orig_h)
                        pad_w = max(0, px - orig_w)
                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            val_img_padded = nn.functional.pad(val_img, (0, pad_w, 0, pad_h, 0, pad_d))
                        else:
                            val_img_padded = val_img

                        with torch.amp.autocast('cuda'):
                            val_out = sliding_window_inference(
                                inputs=val_img_padded,
                                roi_size=tuple(args.patch_size),
                                sw_batch_size=2,
                                predictor=model,
                                overlap=0.5,
                                mode="gaussian",
                            )

                        pred = (torch.softmax(val_out, dim=1)[0, 1, :orig_d, :orig_h, :orig_w] > 0.5).cpu().numpy().astype(np.uint8)
                        inter = np.logical_and(pred, val_lbl).sum()
                        union = pred.sum() + val_lbl.sum()
                        dice = (2.0 * inter / union) if union > 0 else (1.0 if pred.sum() == 0 and val_lbl.sum() == 0 else 0.0)
                        val_dices.append(dice)
                        del val_img, val_img_padded, val_out, pred
                    except Exception as ve:
                        log_print(f"Validation notice: case {cid} skipped due to ({ve})")
                        continue

            torch.cuda.empty_cache()
            mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
            is_best = mean_val_dice > best_val_dice
            if is_best:
                best_val_dice = mean_val_dice

            log_print(f"Validation Epoch {epoch:03d} | Proxy Val Dice: {mean_val_dice:.4f} (Best: {best_val_dice:.4f}) {'[★ NEW BEST]' if is_best else ''}")

            progress_history.append({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_dice": mean_val_dice,
                "lr": current_lr,
                "duration_s": epoch_duration,
            })

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_dice": best_val_dice,
                "progress_history": progress_history,
            }
            torch.save(checkpoint_data, checkpoint_latest)

            if is_best:
                torch.save(checkpoint_data, checkpoint_best)

    total_training_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    log_print(f"=== PRIMUSV2-S TRAINING COMPLETE in {total_training_time}! Best Val Dice: {best_val_dice:.4f} ===")

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": f"PrimusV2-{args.model_id}",
            "fold": args.fold,
            "epochs": args.epochs,
            "best_val_dice": best_val_dice,
            "total_duration": total_training_time,
            "progress": progress_history,
        }, f, indent=2)

if __name__ == "__main__":
    main()
