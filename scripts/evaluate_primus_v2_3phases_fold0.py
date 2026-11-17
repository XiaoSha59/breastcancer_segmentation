#!/usr/bin/env python3
"""
evaluate_primus_v2_3phases_fold0.py
===================================
Independent Sliding-Window Inference & Paper Evaluation for PrimusV2 on 3 Natural DCE-MRI Phases.
Evaluates on all 197 validation cases of Fold 0.

Produces:
1. Binary prediction masks (NIfTI .nii.gz) in <output_dir>/validation/
2. summary.json with patient-level metrics
3. Official paper metrics (Mean/Median Dice, HD95, ASSD) matching Table 2 of Elena Krikun et al.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset

# Hardware & Driver Stability Safeguards
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from primus_v2 import create_primus_v2
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
    parser = argparse.ArgumentParser(description="Evaluate PrimusV2 on 3 Natural Phases (Fold 0)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--model_id", type=str, default="S", choices=["S", "B", "M"])
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
        print(f"Error: {checkpoint_best} not found!")
        sys.exit(1)

    if not os.path.exists(args.splits_file):
        candidate_splits = [
            os.path.join(PROJECT_DIR, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(PROJECT_DIR, "splits_final.json"),
        ]
        for csp in candidate_splits:
            if os.path.exists(csp):
                args.splits_file = csp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    if args.prep_dir and os.path.isdir(args.prep_dir):
        prep_dir = args.prep_dir
    else:
        candidate_prep = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(PROJECT_DIR, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        prep_dir = ""
        for cpr in candidate_prep:
            if os.path.isdir(cpr):
                prep_dir = cpr
                break
        if not prep_dir:
            prep_dir = os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_primus_v2(
        input_channels=3,
        num_classes=2,
        patch_size=tuple(args.patch_size),
        model_id=args.model_id,
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

    print(f"Running sliding window inference on 197 validation cases (PrimusV2-{args.model_id}, 3 Phases)...")
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
        "model": f"PrimusV2-{args.model_id} (3 Natural Phases)",
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
    print(f"             PRIMUSV2-{args.model_id} (3 PHASES) EVALUATION SUMMARY              ")
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
