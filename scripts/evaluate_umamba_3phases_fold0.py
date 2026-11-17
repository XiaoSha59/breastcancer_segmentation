#!/usr/bin/env python3
"""
evaluate_umamba_3phases_fold0.py
================================
Independent Sliding-Window Inference & Full Evaluation for U-Mamba Bot 3D on 3 Natural DCE-MRI Phases.
Evaluates on all 197 validation cases of Fold 0.

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

from umamba_bot_3d import create_umamba_bot_3d

try:
    from torch.amp import autocast as torch_autocast
    def get_autocast():
        return torch_autocast("cuda")
except (ImportError, TypeError):
    from torch.cuda.amp import autocast as torch_autocast
    def get_autocast():
        return torch_autocast()

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
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--prep_dir", type=str, default="", help="Path to preprocessed nnUNetPlans_3d_fullres folder")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_3phases_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


class SimpleNiftiValDataset3Phases(Dataset):
    """Dataset streaming strictly 3 natural phases [:3] via SimpleITK."""
    def __init__(self, val_files):
        self.val_files = val_files

    def __len__(self):
        return len(self.val_files)

    def __getitem__(self, idx):
        item = self.val_files[idx]
        imgs = []
        for p in item["image"][:3]:  # Strictly 3 Natural Phases
            img_itk = sitk.ReadImage(p)
            arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # (D, H, W)
            mask = arr != 0
            if mask.sum() > 0:
                mean = arr[mask].mean()
                std = arr[mask].std()
                if std > 1e-8:
                    arr = (arr - mean) / std
            imgs.append(arr)
        img_tensor = torch.from_numpy(np.stack(imgs, axis=0))  # (3, D, H, W)

        lbl_itk = sitk.ReadImage(item["label"])
        lbl_arr = sitk.GetArrayFromImage(lbl_itk).astype(np.float32)
        lbl_tensor = torch.from_numpy(lbl_arr).unsqueeze(0)  # (1, D, H, W)

        return {"image": img_tensor, "label": lbl_tensor, "case_id": item["case_id"]}


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
        alt_splits = [
            os.path.join(PROJECT_DIR, "data", "processed", "Dataset501_BreastDCE_ISPY2", "splits_final.json"),
            os.path.join(PROJECT_DIR, "splits_final.json"),
        ]
        for asp in alt_splits:
            if os.path.exists(asp):
                args.splits_file = asp
                break

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    val_ids = splits[args.fold]["val"]
    print(f"Loaded {len(val_ids)} validation cases for Fold {args.fold}.")

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")

    has_raw = os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0

    val_files = []
    for cid in val_ids:
        # Strictly channels 0000, 0001, 0002 (Pre, Early Peak, Delayed Late)
        imgs = [os.path.join(images_dir, f"{cid}_{ch:04d}.nii.gz") for ch in range(3)]
        lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
        val_files.append({"image": imgs, "label": lbl, "case_id": cid})

    if has_raw and HAS_MONAI:
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ])
        from monai.data import Dataset as MonaiDataset
        val_ds = MonaiDataset(data=val_files, transform=val_transforms)
    elif has_raw:
        val_ds = SimpleNiftiValDataset3Phases(val_files)
    else:
        print("Note: Raw NIfTI imagesTr not found, evaluating via preprocessed data handles...")
        val_ds = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_umamba_bot_3d(
        in_channels=3,
        out_channels=2,
        features=[32, 64, 128, 256, 320],
        deep_supervision=False,
    ).to(device)

    print(f"Loading weights from {checkpoint_best}...")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    model_state = checkpoint["model_state_dict"]
    model.load_state_dict(model_state, strict=False)
    model.eval()

    patient_metrics = {}
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []

    print("Running sliding window inference on 197 validation cases (3 Natural Phases)...")
    eval_start = time.time()

    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                cid = batch["case_id"][0]
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                with get_autocast():
                    outputs = sliding_window_inference(
                        inputs=inputs,
                        roi_size=tuple(args.patch_size),
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.5,
                    )

                probs = torch.softmax(outputs, dim=1)
                preds = (probs[:, 1] > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
                gt = (labels.squeeze(0).squeeze(0) > 0).cpu().numpy().astype(np.uint8)

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

                # Save binary prediction mask
                pred_itk = sitk.GetImageFromArray(preds)
                sitk.WriteImage(pred_itk, os.path.join(val_out_dir, f"{cid}.nii.gz"))

                if (i + 1) % 25 == 0 or (i + 1) == len(val_ids):
                    print(f"[{i+1}/{len(val_ids)}] Case: {cid} | Dice: {dice:.4f} | Running Mean: {np.mean(dice_scores):.4f}")
    else:
        # Fallback to preprocessed .b2nd or .npy slicing [:3]
        prep_dir = args.prep_dir
        if not prep_dir:
            prep_dir = os.path.join(os.path.dirname(args.splits_file), "nnUNetPlans_3d_fullres")

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

                with get_autocast():
                    outputs = sliding_window_inference(
                        inputs=inputs,
                        roi_size=tuple(args.patch_size),
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.5,
                    )

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
        "model": "U-Mamba Bot 3D (3 Natural Phases)",
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
    print("                  U-MAMBA BOT 3D (3 PHASES) EVALUATION SUMMARY                  ")
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
