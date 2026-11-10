#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

summary_path = Path("/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation/summary.json")

with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== FOREGROUND MEAN ===")
print(json.dumps(data.get("foreground_mean", {}), indent=2))

print("\n=== MEAN PER CLASS ===")
print(json.dumps(data.get("mean", {}), indent=2))

cases = data.get("metric_per_case", [])
print(f"\nTotal Cases: {len(cases)}")

dice_list = []
iou_list = []
fp_list = []
fn_list = []
tp_list = []

for c in cases:
    m = c.get("metrics", {}).get("1", {})
    if "Dice" in m:
        dice_list.append(m["Dice"])
    if "IoU" in m:
        iou_list.append(m["IoU"])
    if "FP" in m:
        fp_list.append(m["FP"])
    if "FN" in m:
        fn_list.append(m["FN"])
    if "TP" in m:
        tp_list.append(m["TP"])

dice_arr = np.array(dice_list)
iou_arr = np.array(iou_list)

print("\n=== STATISTICAL BREAKDOWN (197 CASES) ===")
print(f"Dice Mean     : {np.mean(dice_arr):.4f} +/- {np.std(dice_arr):.4f}")
print(f"Dice Median   : {np.median(dice_arr):.4f}")
print(f"Dice IQR (25-75%): [{np.percentile(dice_arr, 25):.4f}, {np.percentile(dice_arr, 75):.4f}]")
print(f"Dice Min / Max: {np.min(dice_arr):.4f} / {np.max(dice_arr):.4f}")
print(f"IoU Mean      : {np.mean(iou_arr):.4f} +/- {np.std(iou_arr):.4f}")
print(f"IoU Median    : {np.median(iou_arr):.4f}")
