#!/usr/bin/env python3
"""
evaluate_umamba_3phases_fold0.py
=================================
Independent Sliding-Window Inference & Full Evaluation for U-Mamba Bot 3D on Fold 0 (197 Val Cases).
100% faithful clone of evaluate_umamba_fold0.py with strictly 3 Natural DCE-MRI Phases:
1. in_channels: Changed from 5 to 3.
2. Slices strictly 3 Natural Phases (Pre, Early Peak, Delayed Late).
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
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast

# Server node1 hardware safety
torch.backends.cudnn.enabled = False
torch.set_num_threads(4)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from umamba_bot_3d import create_umamba_bot_3d

# Check if MONAI is available; provide fallback if not
try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        NormalizeIntensityd,
        EnsureTyped,
    )
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

    def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.5):
        """Native 3D sliding window inference fallback matching MONAI."""
        B, C, D, H, W = inputs.shape
        r_d, r_h, r_w = roi_size
        step_d = max(1, int(r_d * (1.0 - overlap)))
        step_h = max(1, int(r_h * (1.0 - overlap)))
        step_w = max(1, int(r_w * (1.0 - overlap)))

        d_coords = list(range(0, max(1, D - r_d + 1), step_d))
        if d_coords[-1] + r_d < D:
            d_coords.append(D - r_d)
        h_coords = list(range(0, max(1, H - r_h + 1), step_h))
        if h_coords[-1] + r_h < H:
            h_coords.append(H - r_h)
        w_coords = list(range(0, max(1, W - r_w + 1), step_w))
        if w_coords[-1] + r_w < W:
            w_coords.append(W - r_w)

        output_image = torch.zeros((B, 2, D, H, W), dtype=inputs.dtype, device=inputs.device)
        count_map = torch.zeros((B, 1, D, H, W), dtype=inputs.dtype, device=inputs.device)

        patches = []
        slices = []
        for d in d_coords:
            for h in h_coords:
                for w in w_coords:
                    patch = inputs[:, :, d:d+r_d, h:h+r_h, w:w+r_w]
                    patches.append(patch)
                    slices.append((d, h, w))
                    if len(patches) == sw_batch_size:
                        batch_patches = torch.cat(patches, dim=0)
                        preds = predictor(batch_patches)
                        if isinstance(preds, list):
                            preds = preds[0]
                        for i, (sd, sh, sw) in enumerate(slices):
                            output_image[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += preds[i:i+1]
                            count_map[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += 1.0
                        patches = []
                        slices = []

        if len(patches) > 0:
            batch_patches = torch.cat(patches, dim=0)
            preds = predictor(batch_patches)
            if isinstance(preds, list):
                preds = preds[0]
            for i, (sd, sh, sw) in enumerate(slices):
                output_image[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += preds[i:i+1]
                count_map[:, :, sd:sd+r_d, sh:sh+r_h, sw:sw+r_w] += 1.0

        return output_image / count_map.clamp(min=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Mamba Bot 3D on 3 Natural Phases (Fold 0)")
    parser.add_argument("--prep_dir", type=str, default=None, help="Path to preprocessed nnUNetPlans_3d_fullres dir")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2", help="Path to raw NIfTI dir")
    parser.add_argument("--splits_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    import pickle
    args = parse_args()
    val_out_dir = os.path.join(args.output_dir, "validation")
    gt_out_dir = os.path.join(args.output_dir, "gt_validation")
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    checkpoint_best = os.path.join(args.output_dir, "checkpoint_best.pth")
    if not os.path.exists(checkpoint_best):
        print(f"Error: {checkpoint_best} not found!")
        sys.exit(1)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    splits_file = args.splits_file
    if not splits_file or not os.path.isfile(splits_file):
        candidates = [
            os.path.join(project_root, "splits_final.json"),
            os.path.join(project_root, "data", "preprocessed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
            os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                splits_file = c
                break

    with open(splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold} (3 Natural Phases).")

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

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    has_raw = os.path.isdir(images_dir) and os.path.isdir(labels_dir)
    has_prep = prep_dir and os.path.isdir(prep_dir)

    print(f"Evaluation Data Source: {'Raw NIfTI' if has_raw else ('Preprocessed ' + str(prep_dir) if has_prep else 'Unknown')}")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    model = create_umamba_bot_3d(
        in_channels=3,
        out_channels=2,
        features=[32, 64, 128, 256, 320],
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

    print(f"Running 3-Phase sliding window inference on {len(val_ids)} validation cases...")
    eval_start = time.time()
    pz, py, px = args.patch_size

    with torch.no_grad():
        for i, cid in enumerate(val_ids):
            if has_raw:
                imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(3)]
                lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
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
                    val_arr = np.empty((3, *img_h.shape[1:]), dtype=np.float32)
                    for c in range(3):
                        val_arr[c] = img_h[c]
                    img_raw = val_arr
                    lbl_raw = (np.array(seg_h[0], dtype=np.int16) > 0).astype(np.uint8)
                elif os.path.exists(npy_path):
                    case_data = np.load(npy_path, mmap_mode="r")
                    img_raw = case_data[:3].astype(np.float32)
                    lbl_raw = (case_data[5] > 0).astype(np.uint8)
                else:
                    print(f"Warning: Case {cid} not found, skipping.")
                    continue
            else:
                print(f"Error: Neither preprocessed data nor raw NIfTI found for {cid}!")
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

            pred_np = pred[0, 0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)
            label_np = lbl_raw[:orig_d, :orig_h, :orig_w]

            intersection = np.logical_and(pred_np == 1, label_np == 1).sum()
            union = np.logical_or(pred_np == 1, label_np == 1).sum()
            pred_sum = pred_np.sum()
            label_sum = label_np.sum()

            if pred_sum + label_sum == 0:
                dice, iou, rec, prec = 1.0, 1.0, 1.0, 1.0
            elif pred_sum == 0 or label_sum == 0:
                dice, iou, rec, prec = 0.0, 0.0, (1.0 if label_sum == 0 else 0.0), (1.0 if pred_sum == 0 else 0.0)
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
                "dice": float(dice),
                "iou": float(iou),
                "recall": float(rec),
                "precision": float(prec),
                "pred_voxels": int(pred_sum),
                "gt_voxels": int(label_sum),
            }

            # Save prediction and ground truth NIfTI masks for surface calculation
            pred_sitk = sitk.GetImageFromArray(pred_np)
            gt_sitk = sitk.GetImageFromArray(label_np)

            if has_raw and os.path.exists(os.path.join(labels_dir, f"{cid}.nii.gz")):
                ref_img = sitk.ReadImage(os.path.join(labels_dir, f"{cid}.nii.gz"))
                pred_sitk.SetSpacing(ref_img.GetSpacing())
                pred_sitk.SetOrigin(ref_img.GetOrigin())
                pred_sitk.SetDirection(ref_img.GetDirection())
                gt_sitk.SetSpacing(ref_img.GetSpacing())
                gt_sitk.SetOrigin(ref_img.GetOrigin())
                gt_sitk.SetDirection(ref_img.GetDirection())
            elif prep_dir:
                pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as pf:
                        meta = pickle.load(pf)
                    if "spacing" in meta:
                        sp = meta["spacing"]
                        pred_sitk.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))
                        gt_sitk.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))

            sitk.WriteImage(pred_sitk, os.path.join(val_out_dir, f"{cid}.nii.gz"))
            sitk.WriteImage(gt_sitk, os.path.join(gt_out_dir, f"{cid}.nii.gz"))

            del inputs, outputs, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                print(f"[{i+1}/{len(val_ids)}] {cid}: Dice={dice:.4f} | IoU={iou:.4f} | Running Mean Dice={np.mean(dice_scores):.4f}")

    eval_duration = time.time() - eval_start
    print(f"Inference complete in {eval_duration:.1f}s.")

    summary_data = {
        "model": "U-Mamba Bot 3D (3 Natural Phases)",
        "fold": args.fold,
        "num_cases": len(val_ids),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "median_iou": float(np.median(iou_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "inference_time_seconds": float(eval_duration),
        "cases": patient_metrics,
    }

    # Save summary.json to both val_out_dir and args.output_dir for seamless discovery
    for s_path in [os.path.join(val_out_dir, "summary.json"), os.path.join(args.output_dir, "summary.json")]:
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
    print(f"Saved patient-level summary to {val_out_dir}/summary.json and {args.output_dir}/summary.json")

    # Compute surface metrics (HD95, ASSD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compute_paper_metrics_script = os.path.join(script_dir, "compute_paper_metrics.py")
    if os.path.exists(compute_paper_metrics_script):
        cmd = f"python3 {compute_paper_metrics_script} {args.output_dir} 'U-Mamba Bot 3D 3-Phases (MICCAI 2024)'"
        os.system(cmd)


if __name__ == "__main__":
    main()

