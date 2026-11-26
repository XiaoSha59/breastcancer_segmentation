#!/usr/bin/env python3
"""
evaluate_primus_v2_3phases_fold0.py
====================================
Full Independent Sliding-Window Inference & Paper Evaluation for PrimusV2 on 3 Natural DCE-MRI Phases (Fold 0, 197 Val Cases).
100% faithful clone of evaluate_primus_v2_fold0.py with strictly:
1. input_channels: Changed from 5 to 3.
2. Slices strictly 3 Natural Phases [:3] (Pre, Early Peak, Delayed Late).
3. Produces exact paper metrics (Dice, HD95, ASSD) via compute_paper_metrics.py.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

# Server node1 hardware safety
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from primus_v2 import create_primus_v2
from monai.inferers import sliding_window_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PrimusV2 on 3 Natural Phases (Fold 0)")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 192, 192], help="Optimal patch size [64, 192, 192] or [192, 192, 64]")
    parser.add_argument("--model_id", type=str, default="S", choices=["S", "B", "M"])
    return parser.parse_args()

def main():
    args = parse_args()

    # Auto-align patch_size to (Z, Y, X): e.g. [192, 192, 64] -> [64, 192, 192]
    if args.patch_size[0] > args.patch_size[2]:
        args.patch_size = [args.patch_size[2], args.patch_size[0], args.patch_size[1]]

    # Cluster Safety: Strictly limit CPU threads for shared server (node1)
    torch.set_num_threads(4)

    # Auto-resolve output_dir if default is /home/XiaoSha/... on another machine
    if args.output_dir.startswith("/home/XiaoSha") and not os.path.exists("/home/XiaoSha"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        args.output_dir = os.path.join(project_root, "results", "Dataset501_BreastDCE_ISPY2", f"PrimusV2_{args.model_id}_3phases_200epochs", f"fold_{args.fold}")

    val_out_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        checkpoint_best = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: {checkpoint_best} not found!")
        sys.exit(1)

    # Auto-resolve splits_file if default does not exist
    if not os.path.exists(args.splits_file):
        candidate_splits = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for csp in candidate_splits:
            if os.path.exists(csp):
                args.splits_file = csp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold} (3 Natural Phases).")

    if args.prep_dir and os.path.isdir(args.prep_dir):
        prep_dir = args.prep_dir
    else:
        candidate_prep = [
            os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "nnUNetPlans_3d_fullres"),
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
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # PrimusV2 initialized with input_channels=3
    model = create_primus_v2(
        input_channels=3,
        num_classes=2,
        patch_size=tuple(args.patch_size),
        model_id=args.model_id,
        drop_path_rate=0.0,
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

    print(f"Running sliding window inference on {len(val_ids)} validation cases (3 Natural Phases)...")
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
                img_raw = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                for c in range(3):
                    img_raw[c] = img_h[c]
                lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
            elif os.path.exists(npy_path):
                case_data = np.load(npy_path, mmap_mode="r")
                img_raw = case_data[:3].astype(np.float32)
                lbl_raw = (case_data[5] > 0).astype(np.uint8)
            else:
                print(f"Warning: Case {cid} not found, skipping.")
                continue

            _, orig_d, orig_h, orig_w = img_raw.shape

            # Dynamic padding to patch size
            pad_d = max(0, pz - orig_d)
            pad_h = max(0, py - orig_h)
            pad_w = max(0, px - orig_w)

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img_raw, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:
                img_padded = img_raw

            inputs = torch.from_numpy(img_padded).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda'):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sw_device="cuda",
                    device="cpu",
                )

            prob = torch.softmax(outputs, dim=1)[:, 1:2]
            pred = (prob > 0.5).float()

            pred_np = pred[0, 0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
            label_np = lbl_raw[:orig_d, :orig_h, :orig_w]

            intersection = np.logical_and(pred_np, label_np).sum()
            union = np.logical_or(pred_np, label_np).sum()
            pred_sum = pred_np.sum()
            label_sum = label_np.sum()

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

            # Retrieve physical spacing from .pkl
            pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
            if os.path.exists(pkl_path):
                import pickle
                try:
                    with open(pkl_path, "rb") as pf:
                        meta = pickle.load(pf)
                    raw_sp = meta.get("spacing", [1.0, 1.0, 1.0])
                    spacing = (float(raw_sp[2]), float(raw_sp[1]), float(raw_sp[0]))
                except Exception:
                    spacing = (1.0, 1.0, 1.0)
            else:
                spacing = (1.0, 1.0, 1.0)
            origin = (0.0, 0.0, 0.0)
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            # Save prediction NIfTI
            pred_img = sitk.GetImageFromArray(pred_np)
            pred_img.SetSpacing(spacing)
            pred_img.SetOrigin(origin)
            pred_img.SetDirection(direction)
            sitk.WriteImage(pred_img, os.path.join(val_out_dir, f"{cid}.nii.gz"))

            # Save GT NIfTI for HD95 calculation
            gt_dir_for_eval = os.path.join(val_out_dir, "gt_masks")
            os.makedirs(gt_dir_for_eval, exist_ok=True)
            gt_nii_path = os.path.join(gt_dir_for_eval, f"{cid}.nii.gz")
            if not os.path.exists(gt_nii_path):
                gt_img = sitk.GetImageFromArray(label_np)
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
    print(f"Inference complete in {eval_duration:.1f}s ({eval_duration/len(val_ids):.2f}s/case).")

    summary_data = {
        "model": f"PrimusV2-{args.model_id} (3 Natural Phases)",
        "num_cases": len(val_ids),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "cases": patient_metrics,
        "metric_per_case": [
            {
                "reference_file": f"{cid}.nii.gz",
                "prediction_file": f"{cid}.nii.gz",
                "metrics": {
                    "1": {
                        "Dice": m["Dice"],
                        "IoU": m["IoU"],
                        "Recall": m["Recall"],
                        "Precision": m["Precision"],
                    }
                }
            }
            for cid, m in patient_metrics.items()
        ]
    }

    summary_file = os.path.join(val_out_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Compute official paper surface metrics (HD95, ASSD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compute_paper_metrics_script = os.path.join(script_dir, "compute_paper_metrics.py")
    if os.path.exists(compute_paper_metrics_script):
        print("\nComputing official paper surface metrics (HD95, ASSD)...")
        os.system(f"python3 {compute_paper_metrics_script} {args.output_dir} 'PrimusV2-{args.model_id} 3-Phases (Transformer, DKFZ 2024)'")

if __name__ == "__main__":
    main()
