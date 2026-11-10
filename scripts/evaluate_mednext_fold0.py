#!/usr/bin/env python3
"""
evaluate_mednext_fold0.py
=========================
Independent Sliding-Window Inference & Full Evaluation for MedNeXt on Fold 0 (197 Val Cases).
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mednext_v1 import create_mednext_v1

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
)
from monai.data import Dataset
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedNeXt on I-SPY2 Fold 0")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--model_id", type=str, default="L", choices=["S", "B", "M", "L"])
    parser.add_argument("--kernel_size", type=int, default=3, choices=[3, 5])
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
    model = create_mednext_v1(
        num_channels=5,
        num_classes=2,
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=False,  # In eval mode, only output highest resolution
    ).to(device)

    print(f"Loading weights from {checkpoint_best}...")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    # Filter state dict if deep supervision heads were saved
    model_state = checkpoint["model_state_dict"]
    model.load_state_dict(model_state, strict=False)
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
            cid = batch["case_id"][0]
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                )

            prob = torch.softmax(outputs, dim=1)[:, 1:2]
            pred = (prob > 0.5).float()

            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            label_np = labels.squeeze().cpu().numpy().astype(np.uint8)

            intersection = np.logical_and(pred_np, label_np).sum()
            union = np.logical_or(pred_np, label_np).sum()
            pred_sum = pred_np.sum()
            label_sum = label_np.sum()

            if label_sum == 0 and pred_sum == 0:
                dice, iou, rec, prec = 1.0, 1.0, 1.0, 1.0
            elif label_sum == 0 or pred_sum == 0:
                dice, iou, rec, prec = 0.0, 0.0, (0.0 if label_sum > 0 else 1.0), (0.0 if pred_sum > 0 else 1.0)
            else:
                dice = 2.0 * intersection / (pred_sum + label_sum)
                iou = intersection / union
                rec = intersection / label_sum
                prec = intersection / pred_sum

            dice_scores.append(dice)
            iou_scores.append(iou)
            recall_scores.append(rec)
            precision_scores.append(prec)

            patient_metrics[cid] = {
                "Dice": float(dice),
                "IoU": float(iou),
                "Recall": float(rec),
                "Precision": float(prec),
            }

            ref_path = os.path.join(labels_dir, f"{cid}.nii.gz")
            ref_img = sitk.ReadImage(ref_path)
            pred_img = sitk.GetImageFromArray(pred_np)
            pred_img.SetSpacing(ref_img.GetSpacing())
            pred_img.SetOrigin(ref_img.GetOrigin())
            pred_img.SetDirection(ref_img.GetDirection())
            out_nii_path = os.path.join(val_out_dir, f"{cid}.nii.gz")
            sitk.WriteImage(pred_img, out_nii_path)

            if (i + 1) % 25 == 0 or (i + 1) == len(val_loader):
                print(f"[{i+1}/{len(val_loader)}] {cid}: Dice={dice:.4f} | Running Mean Dice={np.mean(dice_scores):.4f}")

    eval_duration = time.time() - eval_start
    print(f"Inference complete in {eval_duration:.1f}s.")

    summary_data = {
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "cases": patient_metrics,
    }

    summary_file = os.path.join(val_out_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved patient-level summary to {summary_file}")

    # Compute surface metrics (HD95, ASSD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compute_paper_metrics_script = os.path.join(script_dir, "compute_paper_metrics.py")
    if os.path.exists(compute_paper_metrics_script):
        cmd = f"python3 {compute_paper_metrics_script} {args.output_dir} 'MedNeXt-{args.model_id} k{args.kernel_size} (DKFZ)'"
        os.system(cmd)

if __name__ == "__main__":
    main()
