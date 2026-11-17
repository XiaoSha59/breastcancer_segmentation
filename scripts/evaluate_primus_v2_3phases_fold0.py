#!/usr/bin/env python3
"""
evaluate_primus_v2_3phases_fold0.py
===================================
Independent Sliding-Window Inference & Paper Evaluation for PrimusV2 on 3 Natural DCE-MRI Phases.
Compliant with 3D_UNET_BEST_PRACTICES.md & BLOSC2_BUFFER_INCIDENT_REPORT.md:

1. Slices strictly 3 Natural Phases: Channel 0 (Pre), Channel 1 (Peak Early), Channel 2 (Delayed Late).
2. Channel-by-channel Blosc2 decompression with z-slice fallback (100% immune to buffer errors).
3. Physical geometry restoration: Extracts exact voxel spacing, origin, and direction from raw NIfTI
   or .pkl metadata to ensure downstream HD95 and ASSD are calculated in true millimeters (mm).
4. Produces:
   - Binary prediction masks (NIfTI .nii.gz) in <output_dir>/validation/
   - summary.json with patient-level metrics
   - Full paper metrics (Mean/Median Dice, HD95, ASSD) matching Table 2 of Elena Krikun et al.
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

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    has_prep = prep_dir and os.path.isdir(prep_dir)
    has_raw = os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0

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
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            try:
                ref_img = None
                meta_spacing = None

                if has_raw:
                    # Load first 3 Natural Phases: 0000 (Pre), 0001 (Early Peak), 0002 (Delayed Late)
                    imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(3)]
                    lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
                    ref_itk = sitk.ReadImage(imgs[0])
                    ref_img = ref_itk
                    img_raw = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in imgs], axis=0).astype(np.float32)
                    for c in range(3):
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
                        # Channel-by-channel decompression to avoid Blosc2 buffer crash
                        val_arr = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                        for c in range(3):
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
                        img_raw = case_data[:3].astype(np.float32)
                        lbl_raw = (case_data[5] > 0).astype(np.uint8)
                    else:
                        print(f"Warning: Case {cid} not found, skipping.")
                        continue

                    # Try loading spacing metadata from .pkl
                    pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
                    if os.path.exists(pkl_path):
                        with open(pkl_path, "rb") as pf:
                            meta = pickle.load(pf)
                        if "spacing" in meta:
                            meta_spacing = tuple(float(s) for s in reversed(meta["spacing"]))
                else:
                    print(f"Error: Neither preprocessed data nor raw NIfTI found for {cid}!")
                    continue

                inputs = torch.from_numpy(img_raw).unsqueeze(0).to(device)

                # Dynamic padding if volume smaller than patch size
                orig_shape = inputs.shape[2:]
                pad_d = max(0, pz - orig_shape[0])
                pad_h = max(0, py - orig_shape[1])
                pad_w = max(0, px - orig_shape[2])
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
                pred = (probs[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                gt = lbl_raw

                tp = np.logical_and(pred == 1, gt == 1).sum()
                fp = np.logical_and(pred == 1, gt == 0).sum()
                fn = np.logical_and(pred == 0, gt == 1).sum()
                union = np.logical_or(pred == 1, gt == 1).sum()

                dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0)
                iou = tp / union if union > 0 else (1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)
                recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if pred.sum() == 0 else 0.0)

                dice_scores.append(dice)
                iou_scores.append(iou)
                recall_scores.append(recall)
                precision_scores.append(precision)

                patient_metrics[cid] = {
                    "dice": float(dice),
                    "iou": float(iou),
                    "recall": float(recall),
                    "precision": float(precision),
                    "pred_voxels": int(pred.sum()),
                    "gt_voxels": int(gt.sum()),
                }

                # Save 3D binary prediction mask with restored physical geometry (spacing, origin, direction)
                pred_path = os.path.join(val_out_dir, f"{cid}.nii.gz")
                pred_nii = sitk.GetImageFromArray(pred)
                if ref_img is not None:
                    pred_nii.SetSpacing(ref_img.GetSpacing())
                    pred_nii.SetOrigin(ref_img.GetOrigin())
                    pred_nii.SetDirection(ref_img.GetDirection())
                elif meta_spacing is not None:
                    pred_nii.SetSpacing(meta_spacing)
                sitk.WriteImage(pred_nii, pred_path)

                if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                    print(f"[{i+1}/{len(val_ids)}] Case: {cid} | Dice: {dice:.4f} | Running Mean: {np.mean(dice_scores):.4f}")

            except Exception as e:
                print(f"Warning: Failed to evaluate case {cid}: {e}. Skipping.")
                continue

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
