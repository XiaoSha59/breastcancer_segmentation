#!/usr/bin/env python3
"""
evaluate_swinunetr_v2_fold0.py
==============================
Unified & Bulletproof Sliding-Window Inference & Full Evaluation for SwinUNETR V2 on Fold 0 (197 Val Cases).
Compliant with 3D_UNET_BEST_PRACTICES.md & Official nnU-Net Protocol:
1. Evaluates directly on preprocessed nnUNetPlans_3d_fullres (.b2nd / .npy) in resampled model space.
2. Automatically pads volumes if any dimension < patch_size [96, 96, 96], then unpads to original size.
3. Applies Gaussian importance weighting (mode='gaussian') matching nnU-Net's seamless patch fusion.
4. Restores physical geometry headers from metadata (.pkl) to export clean prediction & ground-truth NIfTI masks.
5. Computes Dice, IoU, Precision, Recall and triggers compute_paper_metrics.py to calculate HD95 & ASSD.
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
from torch.cuda.amp import autocast

import monai
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SwinUNETR V2 on I-SPY2 Fold 0")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
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
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: No checkpoint found in {args.output_dir}!")
        sys.exit(1)

    if not os.path.exists(args.splits_file):
        alt_splits = [
            "/home/XiaoSha/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits_final.json"),
        ]
        for asp in alt_splits:
            if os.path.exists(asp):
                args.splits_file = asp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    # Resolve preprocessed data directory
    if args.prep_dir and os.path.isdir(args.prep_dir):
        prep_dir = args.prep_dir
    else:
        candidate_dirs = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres",
            "/home/XiaoSha/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres",
            os.path.join(args.data_dir, "nnUNetPlans_3d_fullres"),
        ]
        prep_dir = ""
        for cd in candidate_dirs:
            if os.path.isdir(cd):
                prep_dir = cd
                break

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    has_prep = prep_dir and os.path.isdir(prep_dir)
    has_raw = os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0

    print(f"Data Source: {'Preprocessed nnUNetPlans_3d_fullres (.b2nd/.npy) [OPTIMAL]' if has_prep else 'Raw NIfTI'}")
    if has_prep:
        print(f"Preprocessed Directory: {prep_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
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
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    patient_metrics = {}
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []

    print(f"Running sliding window inference on {len(val_ids)} validation cases...")
    eval_start = time.time()
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            if has_prep:
                b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
                npy_path = os.path.join(prep_dir, f"{cid}.npy")
                if os.path.exists(b2nd_path):
                    import blosc2
                    img_h = blosc2.open(b2nd_path)
                    seg_h = blosc2.open(os.path.join(prep_dir, f"{cid}_seg.b2nd"))
                    img_raw = np.array(img_h[:], dtype=np.float32)
                    lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                elif os.path.exists(npy_path):
                    case_data = np.load(npy_path, mmap_mode="r")
                    img_raw = case_data[:5].astype(np.float32)
                    lbl_raw = (case_data[5] > 0).astype(np.uint8)
                else:
                    print(f"Warning: Case {cid} not found in {prep_dir}, skipping.")
                    continue
            elif has_raw:
                imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(5)]
                lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
                img_raw = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in imgs], axis=0).astype(np.float32)
                for c in range(5):
                    mask = img_raw[c] != 0
                    if np.any(mask):
                        img_raw[c][mask] = (img_raw[c][mask] - img_raw[c][mask].mean()) / (img_raw[c][mask].std() + 1e-8)
                lbl_raw = (sitk.GetArrayFromImage(sitk.ReadImage(lbl)) > 0).astype(np.uint8)
            else:
                print(f"Error: Neither preprocessed data nor raw NIfTI found for {cid}!")
                continue

            # Original dimensions: C, D, H, W
            _, orig_d, orig_h, orig_w = img_raw.shape

            # Crucial: Pad if any spatial dimension is smaller than patch_size [pz, py, px]
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
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )

            prob = torch.softmax(outputs, dim=1)[:, 1:2]
            pred = (prob > 0.5).float()

            # Safely extract 3D prediction and crop back to original shape
            pred_3d = pred[0, 0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
            label_3d = lbl_raw[:orig_d, :orig_h, :orig_w]

            # Metric computations
            intersection = np.logical_and(pred_3d, label_3d).sum()
            union = np.logical_or(pred_3d, label_3d).sum()
            pred_sum = pred_3d.sum()
            label_sum = label_3d.sum()

            if label_sum == 0 and pred_sum == 0:
                dice, iou, rec, prec = 1.0, 1.0, 1.0, 1.0
            elif pred_sum == 0:
                dice, iou, rec, prec = 0.0, 0.0, 0.0, 0.0
            elif label_sum == 0:
                dice, iou, rec, prec = 0.0, 0.0, 1.0, 0.0
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

            # Physical geometry recovery from metadata (.pkl) or raw image
            spacing = (1.0, 1.0, 1.0)
            origin = (0.0, 0.0, 0.0)
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            if has_raw and os.path.exists(os.path.join(labels_dir, f"{cid}.nii.gz")):
                ref_img = sitk.ReadImage(os.path.join(labels_dir, f"{cid}.nii.gz"))
                spacing = ref_img.GetSpacing()
                origin = ref_img.GetOrigin()
                direction = ref_img.GetDirection()
            elif has_prep:
                pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
                if os.path.exists(pkl_path):
                    try:
                        with open(pkl_path, "rb") as pf:
                            meta = pickle.load(pf)
                        raw_sp = meta.get("spacing", [1.0, 1.0, 1.0])
                        spacing = (float(raw_sp[2]), float(raw_sp[1]), float(raw_sp[0]))
                    except Exception:
                        pass

            pred_img = sitk.GetImageFromArray(pred_3d)
            pred_img.SetSpacing(spacing)
            pred_img.SetOrigin(origin)
            pred_img.SetDirection(direction)
            sitk.WriteImage(pred_img, os.path.join(val_out_dir, f"{cid}.nii.gz"))

            # Also export ground truth mask to validation/gt_masks/ for HD95 / ASSD calculation
            gt_dir_for_eval = os.path.join(val_out_dir, "gt_masks")
            os.makedirs(gt_dir_for_eval, exist_ok=True)
            gt_nii_path = os.path.join(gt_dir_for_eval, f"{cid}.nii.gz")
            if not os.path.exists(gt_nii_path):
                gt_img = sitk.GetImageFromArray(label_3d)
                gt_img.SetSpacing(spacing)
                gt_img.SetOrigin(origin)
                gt_img.SetDirection(direction)
                sitk.WriteImage(gt_img, gt_nii_path)

            del inputs, outputs, prob, pred
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

            if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                print(f"[{i+1}/{len(val_ids)}] {cid}: Dice={dice:.4f} | Running Mean Dice={np.mean(dice_scores):.4f}")

    eval_duration = time.time() - eval_start
    mean_dice = float(np.mean(dice_scores))
    median_dice = float(np.median(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    mean_recall = float(np.mean(recall_scores))
    mean_precision = float(np.mean(precision_scores))

    print("\n" + "=" * 60)
    print(f"SWINUNETR V2 EVALUATION SUMMARY ({len(val_ids)} CASES, FOLD {args.fold})")
    print("=" * 60)
    print(f"Mean Dice:       {mean_dice:.4f} +/- {np.std(dice_scores):.4f}")
    print(f"Median Dice:     {median_dice:.4f}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Recall:     {mean_recall:.4f}")
    print(f"Mean Precision:  {mean_precision:.4f}")
    print(f"Inference Time:  {eval_duration:.1f}s ({eval_duration/len(val_ids):.2f}s/case)")
    print("=" * 60)

    summary_data = {
        "model": "SwinUNETR V2",
        "num_cases": len(val_ids),
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": mean_iou,
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
                        "Dice": m["Dice"],
                        "IoU": m["IoU"],
                        "Precision": m["Precision"],
                        "Recall": m["Recall"],
                    }
                }
            }
            for cid, m in patient_metrics.items()
        ]
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(val_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Compute official paper surface metrics (HD95 & ASSD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_metrics_script = os.path.join(script_dir, "compute_paper_metrics.py")
    if os.path.exists(paper_metrics_script):
        print("\nComputing official paper surface metrics (HD95, ASSD)...")
        os.system(f"python3 {paper_metrics_script} {args.output_dir} 'SwinUNETR V2 (Transformer, MONAI)'")

if __name__ == "__main__":
    main()
