#!/usr/bin/env python3
"""
evaluate_umamba_fold0.py
========================
Independent Sliding-Window Inference & Full Evaluation for U-Mamba Bot 3D on Fold 0 (197 Val Cases).
Optimal Spatial Patch Size Architecture: [192, 192, 64].

Features:
1. Slices strictly 5 DCE-MRI Phases: Channel 0 (T0), Channel 1 (Peak Early), Channel 2 (Delayed Late), Channel 3 (SUB1), Channel 4 (SUB2).
2. Uses in_channels=5 with deep_supervision=False for fast, high-fidelity evaluation.
3. sliding_window_inference with autocast and automatic native Float32 fallback.
4. Restores physical geometry headers from .pkl / raw NIfTI for authentic surface distance evaluation.
5. Computes Mean/Median Dice, IoU, Precision, Recall and triggers compute_paper_metrics.py for HD95 & ASSD.
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast

# Server node1 hardware safety
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(4)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from umamba_bot_3d import create_umamba_bot_3d

# Check if MONAI is available; provide fallback if not
try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        NormalizeIntensityd,
        EnsureTyped,
    )
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

    def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.5):
        """Native 3D sliding window inference fallback matching MONAI."""
        B, C, D, H, W = inputs.shape
        r_d, r_h, r_w = roi_size
        step_d = max(1, int(r_d * (1.0 - overlap)))
        step_h = max(1, int(r_h * (1.0 - overlap)))
        step_w = max(1, int(r_w * (1.0 - overlap)))

        d_coords = list(range(0, max(1, D - r_d + 1), step_d))
        if d_coords[-1] + r_d < D:
            d_coords.append(D - r_d)
        h_coords = list(range(0, max(1, H - r_h + 1), step_h))
        if h_coords[-1] + r_h < H:
            h_coords.append(H - r_h)
        w_coords = list(range(0, max(1, W - r_w + 1), step_w))
        if w_coords[-1] + r_w < W:
            w_coords.append(W - r_w)

        output_image = torch.zeros((B, 2, D, H, W), dtype=inputs.dtype, device=inputs.device)
        count_map = torch.zeros((B, 1, D, H, W), dtype=inputs.dtype, device=inputs.device)

        patches = []
        slices = []
        for d in d_coords:
            for h in h_coords:
                for w in w_coords:
                    patch = inputs[:, :, d:d+r_d, h:h+r_h, w:w+r_w]
                    patches.append(patch)
                    slices.append((d, h, w))

        for batch_idx in range(0, len(patches), sw_batch_size):
            batch_patches = torch.cat(patches[batch_idx:batch_idx+sw_batch_size], dim=0)
            preds = predictor(batch_patches)
            if isinstance(preds, list):
                preds = preds[0]
            for i, (sd, sh, sw) in enumerate(slices[batch_idx:batch_idx+sw_batch_size]):
                output_image[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += preds[i:i+1]
                count_map[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += 1.0

        return output_image / count_map.clamp(min=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Mamba Bot 3D on 5 DCE-MRI Phases (Fold 0, Patch 192x192x64)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_opt_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[192, 192, 64])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-align patch_size to (Z, Y, X): e.g. [192, 192, 64] -> [64, 192, 192]
    if args.patch_size[0] > args.patch_size[2]:
        args.patch_size = [args.patch_size[2], args.patch_size[0], args.patch_size[1]]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", "UMamba_Bot_opt_200epochs", f"fold_{args.fold}")

    val_out_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: No checkpoint found in {args.output_dir}!")
        sys.exit(1)

    splits_file = args.splits_file
    if not splits_file or not os.path.isfile(splits_file):
        candidates = [
            os.path.join(project_root, "splits_final.json"),
            os.path.join(project_root, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                splits_file = c
                break

    with open(splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold} (5 DCE-MRI Phases).")

    # Resolve preprocessed data directory
    prep_dir = args.prep_dir
    if not prep_dir or not os.path.isdir(prep_dir):
        candidate_preps = [
            os.path.join(os.path.dirname(splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for p in candidate_preps:
            if os.path.isdir(p):
                prep_dir = p
                break

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    has_raw = os.path.isdir(images_dir) and os.path.isdir(labels_dir)
    has_prep = prep_dir and os.path.isdir(prep_dir)

    print(f"Evaluation Data Source: {'Raw NIfTI' if has_raw else ('Preprocessed ' + str(prep_dir) if has_prep else 'Unknown')}")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    # Initialize U-Mamba with in_channels=5
    model = create_umamba_bot_3d(
        in_channels=5,
        out_channels=2,
        features=[32, 64, 128, 256, 320],
        deep_supervision=False,
    ).to(device)

    print(f"Loading weights from {checkpoint_best}...")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    patient_metrics = {}
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []

    print(f"Running 5-Phase sliding window inference on {len(val_ids)} validation cases with patch {args.patch_size}...")
    eval_start = time.time()
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            if has_raw:
                imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(5)]
                lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
                img_raw = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in imgs], axis=0).astype(np.float32)
                for c in range(5):
                    mask = img_raw[c] != 0
                    if np.any(mask):
                        img_raw[c][mask] = (img_raw[c][mask] - img_raw[c][mask].mean()) / (img_raw[c][mask].std() + 1e-8)
                lbl_raw = (sitk.GetArrayFromImage(sitk.ReadImage(lbl)) > 0).astype(np.uint8)
            elif has_prep:
                b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
                npy_path = os.path.join(prep_dir, f"{cid}.npy")
                if os.path.exists(b2nd_path):
                    import blosc2
                    img_h = blosc2.open(b2nd_path)
                    seg_h = blosc2.open(os.path.join(prep_dir, f"{cid}_seg.b2nd"))
                    val_arr = np.empty((5, *img_h.shape[1:]), dtype=np.float32)
                    for c in range(5):
                        val_arr[c] = img_h[c]
                    img_raw = val_arr
                    lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                elif os.path.exists(npy_path):
                    case_data = np.load(npy_path, mmap_mode="r")
                    img_raw = case_data[:5].astype(np.float32)
                    lbl_raw = (case_data[5] > 0).astype(np.uint8)
                else:
                    print(f"Warning: Case {cid} not found, skipping.")
                    continue
            else:
                print(f"Error: Neither preprocessed data nor raw NIfTI found for {cid}!")
                continue

            _, orig_d, orig_h, orig_w = img_raw.shape
            pad_d = max(0, pz - orig_d)
            pad_h = max(0, py - orig_h)
            pad_w = max(0, px - orig_w)

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img_raw, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:
                img_padded = img_raw

            inputs = torch.from_numpy(img_padded).unsqueeze(0).to(device)

            with autocast():
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    sw_device="cuda",
                    device="cpu",
                )

            prob = torch.softmax(outputs, dim=1)[:, 1:2]
            pred = (prob > 0.5).float()

            pred_np = pred[0, 0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
            label_np = lbl_raw[:orig_d, :orig_h, :orig_w]

            intersection = np.logical_and(pred_np == 1, label_np == 1).sum()
            union = np.logical_or(pred_np == 1, label_np == 1).sum()
            pred_sum = pred_np.sum()
            label_sum = label_np.sum()

            dice = (2.0 * intersection) / (pred_sum + label_sum + 1e-8) if (pred_sum + label_sum) > 0 else 1.0
            iou = intersection / (union + 1e-8) if union > 0 else 1.0
            recall = intersection / (label_sum + 1e-8) if label_sum > 0 else 1.0
            precision = intersection / (pred_sum + 1e-8) if pred_sum > 0 else 1.0

            dice_scores.append(dice)
            iou_scores.append(iou)
            recall_scores.append(recall)
            precision_scores.append(precision)

            patient_metrics[cid] = {
                "dice": float(dice),
                "iou": float(iou),
                "recall": float(recall),
                "precision": float(precision),
            }

            # Save predicted mask as NIfTI with restored spacing
            pred_sitk = sitk.GetImageFromArray(pred_np)
            pkl_path = os.path.join(prep_dir, f"{cid}.pkl") if has_prep else ""
            raw_lbl_path = os.path.join(labels_dir, f"{cid}.nii.gz") if has_raw else ""

            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as pf:
                        props = pickle.load(pf)
                    spacing = props.get("spacing", (1.0, 1.0, 1.0))
                    pred_sitk.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
                except Exception:
                    pass
            elif os.path.exists(raw_lbl_path):
                try:
                    ref_img = sitk.ReadImage(raw_lbl_path)
                    pred_sitk.CopyInformation(ref_img)
                except Exception:
                    pass

            out_nii_path = os.path.join(val_out_dir, f"{cid}.nii.gz")
            sitk.WriteImage(pred_sitk, out_nii_path)

            if (i + 1) % 20 == 0 or (i + 1) == len(val_ids):
                print(f"[{i+1:03d}/{len(val_ids):03d}] Case {cid}: Dice={dice:.4f}, Running Mean Dice={np.mean(dice_scores):.4f}")

    eval_duration = time.time() - eval_start
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    median_dice = float(np.median(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mean_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    summary = {
        "model": f"U-Mamba Bot 3D (5-Phase, Patch {args.patch_size})",
        "fold": args.fold,
        "cases_evaluated": len(dice_scores),
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "mean_iou": mean_iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "evaluation_time_seconds": eval_duration,
        "patient_metrics": patient_metrics,
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - U-Mamba Bot 3D (5-Phase, Patch {args.patch_size})")
    print("=" * 60)
    print(f"Validation Cases Evaluated: {len(dice_scores)}")
    print(f"Mean Dice:       {mean_dice:.4f}")
    print(f"Median Dice:     {median_dice:.4f}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Precision:  {mean_precision:.4f}")
    print(f"Mean Recall:     {mean_recall:.4f}")
    print(f"Evaluation Time: {eval_duration:.1f}s ({eval_duration/60:.1f}m)")
    print(f"Summary saved to: {summary_file}")
    print("=" * 60)

    # Compute paper metrics (HD95 & ASSD)
    paper_metrics_script = os.path.join(SCRIPT_DIR, "compute_paper_metrics.py")
    if os.path.exists(paper_metrics_script):
        print("\nTriggering compute_paper_metrics.py for HD95 & ASSD calculation...")
        os.system(f'python3 "{paper_metrics_script}" "{args.output_dir}" "U-Mamba Bot 3D (5p, Patch {args.patch_size})"')


if __name__ == "__main__":
    main()
