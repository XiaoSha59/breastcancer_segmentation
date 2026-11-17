#!/usr/bin/env python3
"""
evaluate_segmamba_3phases_fold0.py
==================================
Full Independent Sliding-Window Inference & Paper Evaluation for 3D SegMamba on 3 Natural DCE-MRI Phases (197 Val Cases).
Produces:
1. Binary prediction masks (NIfTI .nii.gz) in <output_dir>/validation/
2. Ground truth masks in <output_dir>/validation/gt_masks/ for Hausdorff calculation
3. summary.json with patient-level metrics
4. Official paper metrics (Mean/Median Dice, HD95, ASSD) matching Elena Krikun et al. Table 2.
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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmamba_3d import create_segmamba_3d
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D SegMamba on 3 Natural Phases (I-SPY2 Fold 0)")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SegMamba_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128])
    return parser.parse_args()

def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.exists(os.path.dirname(args.output_dir)):
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", "SegMamba_3phases_200epochs", f"fold_{args.fold}")

    val_out_dir = os.path.join(args.output_dir, "validation")
    gt_out_dir = os.path.join(val_out_dir, "gt_masks")
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: {checkpoint_best} not found!")
        sys.exit(1)

    if not os.path.exists(args.splits_file):
        candidate_splits = [
            os.path.join(project_root, "splits_final.json"),
            os.path.join(project_root, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for s in candidate_splits:
            if os.path.exists(s):
                args.splits_file = s
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    # Resolve preprocessed data directory
    prep_dir = args.prep_dir
    if not prep_dir or not os.path.isdir(prep_dir):
        candidate_preps = [
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for p in candidate_preps:
            if os.path.isdir(p):
                prep_dir = p
                break

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = create_segmamba_3d(in_channels=3, num_classes=2).to(device)

    print(f"Loading weights from {checkpoint_best}...")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    patient_metrics = {}
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []

    print(f"Running 3-Phase SegMamba sliding window inference on {len(val_ids)} cases...")
    eval_start = time.time()
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
            npy_path = os.path.join(prep_dir, f"{cid}.npy")
            if os.path.exists(b2nd_path):
                import blosc2
                img_h = blosc2.open(b2nd_path)
                seg_h = blosc2.open(os.path.join(prep_dir, f"{cid}_seg.b2nd"))
                # Strictly slice [:3]
                img_raw = np.array(img_h[:3], dtype=np.float32)
                lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
            elif os.path.exists(npy_path):
                case_data = np.load(npy_path, mmap_mode="r")
                img_raw = case_data[:3].astype(np.float32)
                lbl_raw = (case_data[5] > 0).astype(np.uint8)
            else:
                print(f"Warning: Case {cid} not found, skipping.")
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

            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )

            pred_tensor = torch.argmax(outputs, dim=1)
            pred_np = pred_tensor[0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
            label_np = lbl_raw[:orig_d, :orig_h, :orig_w]

            intersection = np.logical_and(pred_np == 1, label_np == 1).sum()
            union = np.logical_or(pred_np, label_np).sum()
            pred_sum = pred_np.sum()
            label_sum = label_np.sum()

            if pred_sum + label_sum == 0:
                dice = 1.0
            elif pred_sum == 0 or label_sum == 0:
                dice = 0.0
            else:
                dice = 2.0 * intersection / (pred_sum + label_sum)

            iou = intersection / union if union > 0 else (1.0 if pred_sum + label_sum == 0 else 0.0)
            precision = intersection / pred_sum if pred_sum > 0 else (1.0 if label_sum == 0 else 0.0)
            recall = intersection / label_sum if label_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)

            patient_metrics[cid] = {
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "tumor_voxels_pred": int(pred_sum),
                "tumor_voxels_gt": int(label_sum),
            }

            # Save prediction and ground truth NIfTI masks for surface calculation
            pred_sitk = sitk.GetImageFromArray(pred_np)
            gt_sitk = sitk.GetImageFromArray(label_np)

            pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    props = pickle.load(f)
                if "spacing" in props:
                    sp = props["spacing"]
                    pred_sitk.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))
                    gt_sitk.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))

            sitk.WriteImage(pred_sitk, os.path.join(val_out_dir, f"{cid}.nii.gz"))
            sitk.WriteImage(gt_sitk, os.path.join(gt_out_dir, f"{cid}.nii.gz"))

            del inputs, outputs, pred_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                print(f"[{i+1}/{len(val_ids)}] Case: {cid} | Dice: {dice:.4f} | IoU: {iou:.4f} | Running Mean Dice: {np.mean(dice_scores):.4f}")

    total_time = time.time() - eval_start
    summary_data = {
        "model": "3D SegMamba (MICCAI 2024, 3 Natural Phases)",
        "fold": args.fold,
        "num_cases": len(val_ids),
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "median_iou": float(np.median(iou_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "inference_time_seconds": float(total_time),
        "cases": patient_metrics,
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 60)
    print(" 3D SEGMAMBA (3 NATURAL PHASES) - VALIDATION RESULTS (FOLD 0)")
    print("=" * 60)
    print(f"Evaluated Cases : {len(dice_scores)} / {len(val_ids)}")
    print(f"Mean Dice       : {np.mean(dice_scores):.4f} +/- {np.std(dice_scores):.4f}")
    print(f"Median Dice     : {np.median(dice_scores):.4f}")
    print(f"Mean IoU        : {np.mean(iou_scores):.4f}")
    print(f"Inference Time  : {total_time:.1f}s")
    print(f"Summary JSON    : {summary_file}")
    print("=" * 60)

    # Automatically compute official paper metrics (HD95 & ASSD)
    compute_metrics_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_paper_metrics.py")
    if os.path.exists(compute_metrics_py):
        print("\nTriggering compute_paper_metrics.py...")
        os.system(f"python3 -u {compute_metrics_py} {args.output_dir} '3D SegMamba 3-Phases (Fold 0, 200 epochs)'")

if __name__ == "__main__":
    main()
