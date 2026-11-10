#!/usr/bin/env python3
"""
evaluate_umamba_fold0.py
========================
Independent Sliding-Window Inference & Full Evaluation for U-Mamba Bot 3D on Fold 0 (197 Val Cases).
Compliant with 3D_UNET_BEST_PRACTICES.md & Official nnU-Net / U-Mamba Evaluation Protocol.

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
from torch.cuda.amp import autocast

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
    parser = argparse.ArgumentParser(description="Evaluate U-Mamba Bot 3D on I-SPY2 Fold 0")
    parser.add_argument("--data_dir", type=str, default="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
    parser.add_argument("--splits_file", type=str, default="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json")
    parser.add_argument("--output_dir", type=str, default="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_200epochs/fold_0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


class SimpleNiftiValDataset(Dataset):
    """Fallback Dataset when MONAI is not available."""
    def __init__(self, val_files):
        self.val_files = val_files

    def __len__(self):
        return len(self.val_files)

    def __getitem__(self, idx):
        item = self.val_files[idx]
        imgs = []
        for p in item["image"]:
            img_itk = sitk.ReadImage(p)
            arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # (D, H, W)
            # Channel-wise Z-score normalization on non-zero voxels
            mask = arr != 0
            if mask.sum() > 0:
                mean = arr[mask].mean()
                std = arr[mask].std()
                if std > 1e-8:
                    arr = (arr - mean) / std
            imgs.append(arr)
        img_tensor = torch.from_numpy(np.stack(imgs, axis=0))  # (5, D, H, W)

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

    if HAS_MONAI:
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ])
        from monai.data import Dataset as MonaiDataset
        val_ds = MonaiDataset(data=val_files, transform=val_transforms)
    else:
        val_ds = SimpleNiftiValDataset(val_files)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_umamba_bot_3d(
        in_channels=5,
        out_channels=2,
        features=[32, 64, 128, 256, 320],
        deep_supervision=False,  # Single high-resolution output during evaluation
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
        cmd = f"python3 {compute_paper_metrics_script} {args.output_dir} 'U-Mamba Bot 3D (MICCAI 2024)'"
        os.system(cmd)


if __name__ == "__main__":
    main()
