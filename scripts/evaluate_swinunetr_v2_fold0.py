#!/usr/bin/env python3
"""
evaluate_swinunetr_v2_fold0.py
==============================
Sliding-Window Inference & Full 197-Case Evaluation for SwinUNETR v2 on 5 DCE-MRI Phases.
Features:
1. Full 5 Phases input: Channel 0 (Pre), 1 (Peak Early), 2 (Delayed Late), 3 (Sub 1), 4 (Sub 2).
2. Uses in_channels=5, out_channels=2, feature_size=48, downsample="mergingv2", use_v2=True.
3. Optimal patch size: 160x160x64 (auto-aligned to (X, Y, Z) array order).
4. Slices strictly 5 channels from Blosc2 (.b2nd) without buffer fragmentation.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SwinUNETR V2 on 5 DCE-MRI Phases (Fold 0)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v2_opt_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[160, 160, 64])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()

    # Auto-align patch_size to match nnU-Net preprocessed array order (X, Y, Z: e.g. 256, 256, 80)
    if args.patch_size[0] < args.patch_size[2]:
        args.patch_size = [args.patch_size[1], args.patch_size[2], args.patch_size[0]]

    # Auto-resolve output_dir if default is /home/XiaoSha/... on another machine
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", "SwinUNETR_v2_opt_200epochs", f"fold_{args.fold}")

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
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres",
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

    print(f"Data Source: {'Raw NIfTI (SimpleITK) [STANDARD - IDENTICAL TO MEDNEXT]' if has_raw else 'Preprocessed nnUNetPlans_3d_fullres'}")
    if has_raw:
        print(f"Raw Images Directory: {images_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Rule 9: Disable cuDNN on node1 for rock-solid stability
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

    model = SwinUNETR(
        in_channels=5,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=False,
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

    print(f"Running 5-Phase sliding window inference on {len(val_ids)} validation cases...")
    eval_start = time.time()
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            try:
                if has_raw:
                    # Load all 5 Phases: 0000 (Pre), 0001 (Early Peak), 0002 (Delayed Late), 0003 (Sub 1), 0004 (Sub 2)
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
                            try:
                                val_arr[c] = img_h[c]
                            except Exception:
                                for z in range(img_h.shape[1]):
                                    val_arr[c, z] = img_h[c, z]
                        img_raw = val_arr
                        try:
                            lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                        except Exception:
                            lbl_raw = np.empty(seg_h.shape[1:], dtype=np.uint8)
                            for z in range(seg_h.shape[1]):
                                lbl_raw[z] = (seg_h[0, z] > 0).astype(np.uint8)
                    elif os.path.exists(npy_path):
                        case_data = np.load(npy_path, mmap_mode="r")
                        img_raw = case_data[:5].astype(np.float32)
                        lbl_raw = (case_data[5] > 0).astype(np.uint8)
                    else:
                        print(f"Warning: Case {cid} not found in {prep_dir}, skipping.")
                        continue
                else:
                    print(f"Error: Neither preprocessed data nor raw NIfTI found for {cid}!")
                    continue

                # Prepare tensor for inference (shape: 1, 5, D, H, W)
                img_tensor = torch.from_numpy(img_raw).unsqueeze(0).to(device)

                # Dynamic sliding window inference
                window_size = (min(pz, img_tensor.shape[2]), min(py, img_tensor.shape[3]), min(px, img_tensor.shape[4]))
                # Ensure window dimensions are divisible by 32 for SwinUNETR v2 hierarchical stages
                window_size = tuple(max(32, (dim // 32) * 32) for dim in window_size)

                logits = sliding_window_inference(
                    inputs=img_tensor,
                    roi_size=window_size,
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian"
                )
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Compute core metrics
                gt = lbl_raw
                intersection = np.logical_and(pred == 1, gt == 1).sum()
                union = np.logical_or(pred == 1, gt == 1).sum()
                pred_sum = (pred == 1).sum()
                gt_sum = (gt == 1).sum()

                if pred_sum + gt_sum == 0:
                    dice = 1.0
                elif pred_sum == 0 or gt_sum == 0:
                    dice = 0.0
                else:
                    dice = 2.0 * intersection / (pred_sum + gt_sum)

                iou = intersection / union if union > 0 else (1.0 if pred_sum + gt_sum == 0 else 0.0)
                precision = intersection / pred_sum if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
                recall = intersection / gt_sum if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

                patient_metrics[cid] = {
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "pred_voxels": int(pred_sum),
                    "gt_voxels": int(gt_sum)
                }

                # Save 3D binary prediction mask for downstream HD95/ASSD
                pred_path = os.path.join(val_out_dir, f"{cid}.nii.gz")
                pred_nii = sitk.GetImageFromArray(pred)
                # Recover spacing & origin if raw NIfTI or .pkl available
                raw_ref = os.path.join(images_dir, f"{cid}_0000.nii.gz")
                if os.path.exists(raw_ref):
                    ref_img = sitk.ReadImage(raw_ref)
                    pred_nii.SetSpacing(ref_img.GetSpacing())
                    pred_nii.SetOrigin(ref_img.GetOrigin())
                    pred_nii.SetDirection(ref_img.GetDirection())
                elif prep_dir:
                    pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
                    if os.path.exists(pkl_path):
                        with open(pkl_path, "rb") as pf:
                            meta = pickle.load(pf)
                        if "spacing" in meta:
                            pred_nii.SetSpacing(tuple(float(s) for s in reversed(meta["spacing"])))
                sitk.WriteImage(pred_nii, pred_path)

                if (i + 1) % 10 == 0 or (i + 1) == len(val_ids):
                    elapsed = time.time() - eval_start
                    print(f"[{i+1}/{len(val_ids)}] Case: {cid} | Dice: {dice:.4f} | IoU: {iou:.4f} | Running Mean Dice: {np.mean(dice_scores):.4f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"Error evaluating {cid}: {str(e)}")
                import traceback
                traceback.print_exc()

    # Summary Statistics
    summary = {
        "model": "SwinUNETR_v2_opt_200epochs",
        "fold": args.fold,
        "evaluated_cases": len(dice_scores),
        "total_val_cases": len(val_ids),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "median_iou": float(np.median(iou_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "patient_metrics": patient_metrics
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    val_summary_file = os.path.join(val_out_dir, "summary.json")
    with open(val_summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SWINUNETR V2 (5-PHASE OPTIMAL) EVALUATION FINISHED")
    print("=" * 60)
    print(f"Evaluated Cases: {len(dice_scores)}/{len(val_ids)}")
    print(f"Mean Dice:       {np.mean(dice_scores):.4f}")
    print(f"Median Dice:     {np.median(dice_scores):.4f}")
    print(f"Mean IoU:        {np.mean(iou_scores):.4f}")
    print(f"Mean Precision:  {np.mean(precision_scores):.4f}")
    print(f"Mean Recall:     {np.mean(recall_scores):.4f}")
    print(f"Results saved to: {summary_file}")
    print("=" * 60)

    # Automatically trigger compute_paper_metrics.py to calculate HD95 and ASSD
    compute_metrics_script = os.path.join(SCRIPT_DIR, "compute_paper_metrics.py")
    if os.path.exists(compute_metrics_script):
        print("\nTriggering compute_paper_metrics.py for HD95 and ASSD...")
        import subprocess
        subprocess.run([
            sys.executable,
            compute_metrics_script,
            "--pred_dir", val_out_dir,
            "--gt_dir", labels_dir if has_raw else prep_dir,
            "--output_file", os.path.join(args.output_dir, "paper_metrics_summary.txt")
        ])

if __name__ == "__main__":
    main()
