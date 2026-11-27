#!/usr/bin/env python3
"""
evaluate_mednext_fold0.py
=========================
Sliding-Window Inference & Full 197-Case Evaluation for MedNeXt-L k3 on 5 DCE-MRI Phases.
Optimal Spatial Patch Size: [192, 192, 64].

Features:
1. Slices strictly 5 DCE-MRI Phases: Channel 0 (T0), Channel 1 (Peak Early), Channel 2 (Delayed Late), Channel 3 (SUB1), Channel 4 (SUB2).
2. Uses num_channels=5 with deep_supervision=False for fast, high-fidelity evaluation.
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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from scripts.mednext_v1 import create_mednext_v1
except ImportError:
    from mednext_v1 import create_mednext_v1

from monai.inferers import sliding_window_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedNeXt-L k3 on 5 DCE-MRI Phases (Fold 0, Patch 192x192x64)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_opt_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[192, 192, 64])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="L")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-align patch_size to (Z, Y, X): e.g. [192, 192, 64] -> [64, 192, 192]
    if args.patch_size[0] > args.patch_size[2]:
        args.patch_size = [args.patch_size[2], args.patch_size[0], args.patch_size[1]]

    # Cluster Safety: Strictly limit CPU threads for shared server (node1)
    torch.set_num_threads(4)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", f"MedNeXt_{args.model_id}_k{args.kernel_size}_opt_200epochs", f"fold_{args.fold}")

    val_out_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: No checkpoint found in {args.output_dir}!")
        sys.exit(1)

    # Auto-resolve splits_file
    if not args.splits_file or not os.path.exists(args.splits_file):
        alt_splits = [
            os.path.join(project_root, "splits_final.json"),
            os.path.join(project_root, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for asp in alt_splits:
            if os.path.exists(asp):
                args.splits_file = asp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded split fold {args.fold}: {len(val_ids)} validation cases.")

    # Auto-resolve prep_dir
    prep_dir = args.prep_dir
    if not prep_dir or not os.path.isdir(prep_dir):
        alt_preps = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        ]
        for apr in alt_preps:
            if os.path.isdir(apr):
                prep_dir = apr
                break

    # Auto-resolve raw data_dir
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        alt_data = [
            os.path.join(project_root, "data", "processed", "Dataset501_BreastDCE_ISPY2"),
            os.path.expanduser("~/breastcancer_segmentation/data/processed/Dataset501_BreastDCE_ISPY2"),
            os.path.expanduser("~/data/processed/Dataset501_BreastDCE_ISPY2"),
        ]
        for adr in alt_data:
            if os.path.isdir(adr):
                data_dir = adr
                break

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    has_raw = os.path.isdir(images_dir) and os.path.isdir(labels_dir)
    has_prep = prep_dir and os.path.isdir(prep_dir)

    print(f"Data source - Raw NIfTI: {has_raw} ({images_dir}), Preprocessed: {has_prep} ({prep_dir})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize MedNeXt with strictly num_channels=5
    model = create_mednext_v1(
        num_channels=5,
        num_classes=2,
        model_id=args.model_id,
        kernel_size=args.kernel_size,
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

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            try:
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
                        val_buf = np.empty((5, *img_h.shape[1:]), dtype=np.float32)
                        for c in range(5):
                            val_buf[c] = img_h[c]
                        img_raw = val_buf
                        lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
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

                input_tensor = torch.from_numpy(img_raw).unsqueeze(0).to(device)

                # Sliding window inference
                try:
                    with torch.amp.autocast('cuda'):
                        output = sliding_window_inference(
                            inputs=input_tensor,
                            roi_size=tuple(args.patch_size),
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                            sw_device="cuda",
                            device="cpu",
                        )
                except Exception:
                    output = sliding_window_inference(
                        inputs=input_tensor,
                        roi_size=tuple(args.patch_size),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        sw_device="cuda",
                        device="cpu",
                    )

                if isinstance(output, list):
                    output = output[0]

                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                # Metrics calculation
                intersection = np.sum((pred == 1) & (lbl_raw == 1))
                total_pred = np.sum(pred == 1)
                total_gt = np.sum(lbl_raw == 1)

                dice = (2.0 * intersection) / (total_pred + total_gt + 1e-8) if (total_pred + total_gt) > 0 else 1.0
                iou = intersection / (total_pred + total_gt - intersection + 1e-8) if (total_pred + total_gt - intersection) > 0 else 1.0
                precision = intersection / (total_pred + 1e-8) if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
                recall = intersection / (total_gt + 1e-8) if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)

                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

                patient_metrics[cid] = {
                    "Dice": float(dice),
                    "IoU": float(iou),
                    "Precision": float(precision),
                    "Recall": float(recall),
                }

                # Save predicted NIfTI mask with authentic physical header
                pred_sitk = sitk.GetImageFromArray(pred)
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

            except Exception as e:
                print(f"Error evaluating case {cid}: {e}")
                continue

    eval_duration = time.time() - eval_start
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    median_dice = float(np.median(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mean_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    summary = {
        "model": f"MedNeXt-{args.model_id} k{args.kernel_size} (5-Phase, Patch {args.patch_size})",
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
    print(f"EVALUATION SUMMARY - MedNeXt-{args.model_id} k{args.kernel_size} (5-Phase, Patch {args.patch_size})")
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
        os.system(f'python3 "{paper_metrics_script}" "{args.output_dir}" "MedNeXt-{args.model_id} k{args.kernel_size} (5p, Patch {args.patch_size})"')


if __name__ == "__main__":
    main()
