#!/usr/bin/env python3
"""
evaluate_swinunetr_3phases_fold0.py
===================================
Sliding-Window Inference & Full 197-Case Evaluation for SwinUNETR V1 on 3 Natural DCE-MRI Phases.
Compliant with 3D_UNET_BEST_PRACTICES.md & Official MONAI protocol:

1. Slices strictly 3 Natural Phases: Channel 0 (Pre), Channel 1 (Peak Early), Channel 2 (Delayed Late).
2. Uses in_channels=3, out_channels=2, feature_size=48, spatial_dims=3 (Standard SwinUNETR V1).
3. Evaluates directly on preprocessed nnUNetPlans_3d_fullres (.b2nd / .npy) or raw NIfTI with dynamic padding/unpadding.
4. Computes Mean/Median Dice, IoU, Precision, Recall and exports summary.json + binary prediction masks.
5. Fully compatible with compute_paper_metrics.py to calculate official HD95 & ASSD.
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

# Hardware & Driver Stability Safeguards
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import monai
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

try:
    from torch.amp import autocast as torch_autocast
    def get_autocast():
        return torch_autocast("cuda")
except (ImportError, TypeError):
    from torch.cuda.amp import autocast as torch_autocast
    def get_autocast():
        return torch_autocast()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SwinUNETR V1 on 3 Natural DCE-MRI Phases (Fold 0)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v1_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    val_out_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: No checkpoint found in {args.output_dir}!")
        sys.exit(1)

    if not os.path.exists(args.splits_file):
        alt_splits = [
            os.path.join(PROJECT_DIR, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(PROJECT_DIR, "splits_final.json"),
        ]
        for asp in alt_splits:
            if os.path.exists(asp):
                args.splits_file = asp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    if args.prep_dir and os.path.isdir(args.prep_dir):
        prep_dir = args.prep_dir
    else:
        candidate_dirs = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(PROJECT_DIR, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        prep_dir = ""
        for cd in candidate_dirs:
            if os.path.isdir(cd):
                prep_dir = cd
                break
        if not prep_dir:
            prep_dir = os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres")

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    has_prep = prep_dir and os.path.isdir(prep_dir)
    has_raw = os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SwinUNETR V1 with in_channels=3
    model = SwinUNETR(
        in_channels=3,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    print(f"Loading weights from {checkpoint_best}...")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patient_metrics = {}
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []

    print(f"Running sliding window inference on 197 validation cases (SwinUNETR V1, 3 Phases)...")
    eval_start = time.time()

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
            npy_path = os.path.join(prep_dir, f"{cid}.npy")

            if os.path.exists(b2nd_path):
                import blosc2
                img_h = blosc2.open(b2nd_path, mode="r")
                seg_h = blosc2.open(os.path.join(prep_dir, f"{cid}_seg.b2nd"), mode="r")
                inputs = torch.from_numpy(np.array(img_h[:3], dtype=np.float32)).unsqueeze(0).to(device)
                gt = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
            elif os.path.exists(npy_path):
                data = np.load(npy_path, mmap_mode="r")
                inputs = torch.from_numpy(data[:3].astype(np.float32)).unsqueeze(0).to(device)
                gt = (data[5] > 0).astype(np.uint8)
            elif has_raw:
                # Fallback to reading 3 raw NIfTI files
                imgs = []
                for ch in range(3):
                    p = os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz")
                    img_itk = sitk.ReadImage(p)
                    arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)
                    mask = arr != 0
                    if mask.sum() > 0:
                        mean = arr[mask].mean()
                        std = arr[mask].std()
                        if std > 1e-8:
                            arr = (arr - mean) / std
                    imgs.append(arr)
                inputs = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).to(device)
                lbl_itk = sitk.ReadImage(os.path.join(labels_dir, f"{cid}.nii.gz"))
                gt = (sitk.GetArrayFromImage(lbl_itk) > 0).astype(np.uint8)
            else:
                continue

            orig_shape = inputs.shape[2:]
            pad_d = max(0, args.patch_size[0] - orig_shape[0])
            pad_h = max(0, args.patch_size[1] - orig_shape[1])
            pad_w = max(0, args.patch_size[2] - orig_shape[2])
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d))

            with get_autocast():
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                )

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

            probs = torch.softmax(outputs, dim=1)
            preds = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)

            tp = np.logical_and(preds == 1, gt == 1).sum()
            fp = np.logical_and(preds == 1, gt == 0).sum()
            fn = np.logical_and(preds == 0, gt == 1).sum()

            dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if (preds.sum() == 0 and gt.sum() == 0) else 0.0)
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if (preds.sum() == 0 and gt.sum() == 0) else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if preds.sum() == 0 else 0.0)

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

            pred_itk = sitk.GetImageFromArray(preds)
            sitk.WriteImage(pred_itk, os.path.join(val_out_dir, f"{cid}.nii.gz"))

            if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                print(f"[{i+1}/{len(val_ids)}] Case: {cid} | Dice: {dice:.4f} | Running Mean: {np.mean(dice_scores):.4f}")

    eval_duration = time.time() - eval_start
    summary = {
        "model": "SwinUNETR V1 (3 Natural Phases)",
        "fold": args.fold,
        "n_cases": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "evaluation_time_seconds": eval_duration,
        "patient_metrics": patient_metrics,
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("================================================================================")
    print("             SWINUNETR V1 (3 PHASES) EVALUATION SUMMARY                         ")
    print("================================================================================")
    print(f"Evaluated Cases : {len(dice_scores)} / {len(val_ids)}")
    print(f"Mean Dice       : {summary['mean_dice']:.4f} +/- {summary['std_dice']:.4f}")
    print(f"Median Dice     : {summary['median_dice']:.4f}")
    print(f"Mean IoU        : {summary['mean_iou']:.4f}")
    print(f"Mean Recall     : {summary['mean_recall']:.4f}")
    print(f"Mean Precision  : {summary['mean_precision']:.4f}")
    print(f"Evaluation Time : {eval_duration:.1f}s")
    print(f"Results Saved   : {summary_file}")
    print("================================================================================")


if __name__ == "__main__":
    main()
