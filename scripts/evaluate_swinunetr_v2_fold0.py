#!/usr/bin/env python3
"""
evaluate_swinunetr_v2_fold0.py
==============================
Independent Sliding-Window Inference & Full Evaluation for SwinUNETR V2 on Fold 0 (197 Val Cases).
Produces:
1. Binary prediction masks (NIfTI .nii.gz) in <output_dir>/validation/
2. summary.json containing patient-level Dice, Recall, Precision, IoU
3. Full paper metrics (Mean/Median Dice, HD95, ASSD) matching Table 2 of Elena Krikun et al.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import monai
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
)
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SwinUNETR V2 on I-SPY2 Fold 0")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v2_200epochs/fold_0")
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
        print(f"Error: {checkpoint_best} not found!")
        sys.exit(1)

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")

    val_files = []
    for cid in val_ids:
        imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(5)]
        lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
        val_files.append({"image": imgs, "label": lbl, "case_id": cid})

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SwinUNETR V2 model definition
    model = SwinUNETR(
        img_size=tuple(args.patch_size),
        in_channels=5,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=True,
        spatial_dims=3,
        downsample="mergingv2",
        use_v2=True,
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

    print("Running sliding window inference on 197 validation cases...")
    eval_start = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            case_id = batch["case_id"][0]
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                val_logits = sliding_window_inference(
                    inputs=images,
                    roi_size=args.patch_size,
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
                pred = torch.argmax(val_logits, dim=1, keepdim=True)

            # Binary metrics
            pred_fg = (pred == 1)
            gt_fg = (labels == 1)

            tp = (pred_fg & gt_fg).sum().float().item()
            fp = (pred_fg & (~gt_fg)).sum().float().item()
            fn = ((~pred_fg) & gt_fg).sum().float().item()

            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)

            dice_scores.append(dice)
            iou_scores.append(iou)
            recall_scores.append(recall)
            precision_scores.append(precision)

            patient_metrics[case_id] = {
                "dice": dice,
                "iou": iou,
                "recall": recall,
                "precision": precision,
            }

            # Save prediction as NIfTI (.nii.gz) for surface metric evaluation
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            pred_sitk = sitk.GetImageFromArray(pred_np)

            # Copy spacing and direction from original ground truth if available
            gt_nii_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
            if os.path.exists(gt_nii_path):
                orig_img = sitk.ReadImage(gt_nii_path)
                pred_sitk.SetSpacing(orig_img.GetSpacing())
                pred_sitk.SetOrigin(orig_img.GetOrigin())
                pred_sitk.SetDirection(orig_img.GetDirection())

            out_nii = os.path.join(val_out_dir, f"{case_id}.nii.gz")
            sitk.WriteImage(pred_sitk, out_nii)

            if (i + 1) % 25 == 0 or (i + 1) == len(val_loader):
                elapsed = time.time() - eval_start
                print(f"[{i+1}/{len(val_loader)}] Evaluated {case_id} | Running Mean Dice: {np.mean(dice_scores):.4f} | Time: {elapsed:.1f}s")

    eval_duration = time.time() - eval_start

    # Summary metrics
    mean_dice = float(np.mean(dice_scores))
    median_dice = float(np.median(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    median_iou = float(np.median(iou_scores))
    mean_recall = float(np.mean(recall_scores))
    mean_precision = float(np.mean(precision_scores))

    print("\n" + "=" * 60)
    print(f"SWINUNETR V2 EVALUATION SUMMARY (197 CASES, FOLD {args.fold})")
    print("=" * 60)
    print(f"Mean Dice:       {mean_dice:.4f} +/- {np.std(dice_scores):.4f}")
    print(f"Median Dice:     {median_dice:.4f}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Recall:     {mean_recall:.4f}")
    print(f"Mean Precision:  {mean_precision:.4f}")
    print(f"Inference Time:  {eval_duration:.1f}s ({eval_duration/len(val_loader):.2f}s/case)")
    print("=" * 60)

    summary_data = {
        "model": "SwinUNETR V2",
        "num_cases": len(val_loader),
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "eval_time_seconds": eval_duration,
        "patient_metrics": patient_metrics,
        "metric_per_case": [
            {
                "reference_file": f"{cid}.nii.gz",
                "prediction_file": f"{cid}.nii.gz",
                "metrics": {
                    "1": {
                        "Dice": m["dice"],
                        "IoU": m["iou"],
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                    }
                }
            }
            for cid, m in patient_metrics.items()
        ]
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(val_out_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # Compute HD95 & ASSD using project's compute_paper_metrics.py if present
    compute_paper_metrics_script = "/home/XiaoSha/project/scripts/compute_paper_metrics.py"
    if os.path.exists(compute_paper_metrics_script):
        print("\nComputing official paper surface metrics (HD95, ASSD)...")
        cmd = f"python3 {compute_paper_metrics_script} {args.output_dir} 'SwinUNETR V2 (MONAI)'"
        os.system(cmd)

if __name__ == "__main__":
    main()
