#!/usr/bin/env python3
"""
eval_missing_case.py
====================
Slices and evaluates any oversized Blosc2 case (e.g. ISPY2-987995) slice-by-slice along Z-axis.
Bypasses the C-Blosc2 800MB contiguous buffer limit by decompressing 1MB slices.
Merges the result into summary.json and recalculates official paper metrics for 197/197 cases.
"""

import os
import sys
import json
import torch
import numpy as np
import SimpleITK as sitk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import monai
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "ISPY2-987995"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v2_200epochs/fold_0"
    prep_dir = sys.argv[3] if len(sys.argv) > 3 else "data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"

    print(f"=== EVALUATING OVERSIZED CASE: {cid} ===")
    val_out_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_out_dir, exist_ok=True)

    b2nd_path = os.path.join(prep_dir, f"{cid}.b2nd")
    seg_path = os.path.join(prep_dir, f"{cid}_seg.b2nd")

    if not os.path.exists(b2nd_path):
        print(f"Error: {b2nd_path} not found!")
        sys.exit(1)

    import blosc2
    img_h = blosc2.open(b2nd_path)
    seg_h = blosc2.open(seg_path)

    n_chans = min(5, img_h.shape[0])
    orig_d, orig_h, orig_w = img_h.shape[1:]
    print(f"Case shape: {n_chans} channels x {orig_d} x {orig_h} x {orig_w}")
    print("Decompressing slice-by-slice along Z-axis (1MB per slice, 0 buffer risk)...")

    val_arr = np.empty((n_chans, orig_d, orig_h, orig_w), dtype=np.float32)
    for c in range(n_chans):
        for z in range(orig_d):
            val_arr[c, z] = img_h[c, z]

    lbl_arr = np.empty((orig_d, orig_h, orig_w), dtype=np.uint8)
    for z in range(orig_d):
        lbl_arr[z] = (np.array(seg_h[0, z], dtype=np.int16) > 0).astype(np.uint8)

    print("Data decompression complete! Initializing SwinUNETR V2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        in_channels=5,
        out_channels=2,
        feature_size=48,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="mergingv2",
        use_v2=True,
    ).to(device)

    checkpoint_best = os.path.join(output_dir, "checkpoint_best.pth")
    checkpoint = torch.load(checkpoint_best, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    pz, py, px = 96, 96, 96
    pad_d = max(0, pz - orig_d)
    pad_h = max(0, py - orig_h)
    pad_w = max(0, px - orig_w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        img_padded = np.pad(val_arr, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    else:
        img_padded = val_arr

    inputs = torch.from_numpy(img_padded).unsqueeze(0).to(device).contiguous()

    print(f"Running Sliding Window Inference on {cid}...")
    with torch.no_grad():
        try:
            with torch.amp.autocast('cuda'):
                outputs = sliding_window_inference(inputs, (96, 96, 96), 2, model, 0.5)
        except Exception:
            outputs = sliding_window_inference(inputs, (96, 96, 96), 2, model, 0.5)

    prob = torch.softmax(outputs, dim=1)[:, 1:2]
    pred = (prob > 0.5).float()
    pred_3d = pred[0, 0, :orig_d, :orig_h, :orig_w].cpu().numpy().astype(np.uint8)

    inter = np.logical_and(pred_3d, lbl_arr).sum()
    union = pred_3d.sum() + lbl_arr.sum()
    dice = float(2.0 * inter / union) if union > 0 else (1.0 if pred_3d.sum() == 0 and lbl_arr.sum() == 0 else 0.0)
    iou = float(inter / (np.logical_or(pred_3d, lbl_arr).sum() + 1e-8))
    rec = float(inter / (lbl_arr.sum() + 1e-8))
    prec = float(inter / (pred_3d.sum() + 1e-8))
    print(f"==> RESULT FOR {cid}: Dice = {dice:.4f}, IoU = {iou:.4f}, Recall = {rec:.4f}, Precision = {prec:.4f}")

    # Geometry recovery
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    pkl_path = os.path.join(prep_dir, f"{cid}.pkl")
    if os.path.exists(pkl_path):
        import pickle
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

    gt_dir_for_eval = os.path.join(val_out_dir, "gt_masks")
    os.makedirs(gt_dir_for_eval, exist_ok=True)
    gt_img = sitk.GetImageFromArray(lbl_arr)
    gt_img.SetSpacing(spacing)
    gt_img.SetOrigin(origin)
    gt_img.SetDirection(direction)
    sitk.WriteImage(gt_img, os.path.join(gt_dir_for_eval, f"{cid}.nii.gz"))

    # Update summary.json
    summary_path = os.path.join(val_out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s_data = json.load(f)

        s_data["patient_metrics"][cid] = {
            "Dice": dice,
            "IoU": iou,
            "Recall": rec,
            "Precision": prec,
        }

        # Check if already in metric_per_case
        existing_refs = [c.get("reference_file") for c in s_data.get("metric_per_case", [])]
        if f"{cid}.nii.gz" not in existing_refs:
            s_data["metric_per_case"].append({
                "reference_file": f"{cid}.nii.gz",
                "prediction_file": f"{cid}.nii.gz",
                "metrics": {
                    "1": {
                        "Dice": dice,
                        "IoU": iou,
                        "Precision": prec,
                        "Recall": rec,
                    }
                }
            })

        all_dices = [m["Dice"] for m in s_data["patient_metrics"].values()]
        s_data["num_cases"] = len(all_dices)
        s_data["mean_dice"] = float(np.mean(all_dices))
        s_data["median_dice"] = float(np.median(all_dices))
        s_data["std_dice"] = float(np.std(all_dices))

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(s_data, f, indent=2)
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s_data, f, indent=2)

        print(f"🎉 Updated summary.json! Total evaluated cohort: {s_data['num_cases']}/197 cases.")
        print(f"Updated Overall Mean Dice: {s_data['mean_dice']:.4f} +/- {s_data['std_dice']:.4f}")
        print(f"Updated Overall Median Dice: {s_data['median_dice']:.4f}")

    # Trigger compute_paper_metrics.py
    paper_metrics_script = os.path.join(SCRIPT_DIR, "compute_paper_metrics.py")
    if os.path.exists(paper_metrics_script):
        print("\nRecalculating official paper surface metrics (HD95, ASSD) for 197/197 cases...")
        os.system(f"python3 {paper_metrics_script} {output_dir} 'SwinUNETR V2 (Transformer, MONAI)'")

if __name__ == "__main__":
    main()
