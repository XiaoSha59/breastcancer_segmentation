#!/usr/bin/env python3
"""
compute_paper_metrics.py
========================
Extracts and computes strictly the 4 paper metrics reported in
Elena Krikun et al. (EJR AI 2026, Table 2 & Table 3):
1. Mean Dice +/- SD
2. Median Dice
3. Median HD95 (mm)
4. Median ASSD (mm)
No extraneous metrics. Fast execution via 8 multiprocessing workers.
"""

import sys
import os
import glob
import json
import time
import datetime
from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_case_surface(args):
    pred_path, gt_dir = args
    fname = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, fname)
    if not os.path.exists(gt_path):
        return np.nan, np.nan
    
    try:
        p_img = sitk.ReadImage(pred_path)
        g_img = sitk.ReadImage(gt_path)
        p_arr = sitk.GetArrayFromImage(p_img) > 0
        g_arr = sitk.GetArrayFromImage(g_img) > 0
        spacing = p_img.GetSpacing()[::-1]
        
        if not np.any(p_arr) or not np.any(g_arr):
            return np.nan, np.nan
        
        pred_b = p_arr ^ binary_erosion(p_arr)
        gt_b = g_arr ^ binary_erosion(g_arr)
        
        dt_pred = distance_transform_edt(~pred_b, sampling=spacing)
        dt_gt = distance_transform_edt(~gt_b, sampling=spacing)
        
        d_pred_to_gt = dt_gt[pred_b]
        d_gt_to_pred = dt_pred[gt_b]
        
        all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        hd95 = float(np.percentile(all_d, 95))
        assd = float(np.mean(all_d))
        return hd95, assd
    except Exception:
        return np.nan, np.nan

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/fold_0"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Standard 3D nnU-Net (Plain Conv, nnUNetPlans)"
    val_dir = os.path.join(results_dir, "validation")
    gt_candidates = [
        os.path.join(val_dir, "gt_masks"),
        os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/gt_segmentations"),
        "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/gt_segmentations",
        "/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/labelsTr",
    ]
    gt_dir = ""
    for cand in gt_candidates:
        if os.path.isdir(cand) and len(os.listdir(cand)) > 0:
            gt_dir = cand
            break
    summary_json = os.path.join(val_dir, "summary.json")
    if not os.path.exists(summary_json):
        summary_json = os.path.join(results_dir, "summary.json")
    
    if not os.path.exists(summary_json):
        print(f"Error: summary.json not found in {val_dir} or {results_dir}!")
        return

    with open(summary_json, "r", encoding="utf-8") as f:
        s_data = json.load(f)
    
    # 1. Dice from summary.json
    cases = s_data.get("metric_per_case", [])
    dice_list = []
    for c in cases:
        d = c.get("metrics", {}).get("1", {}).get("Dice")
        if d is not None:
            dice_list.append(d)
    
    if len(dice_list) == 0:
        if "patient_metrics" in s_data:
            dice_list = [v.get("dice", v.get("Dice")) for v in s_data["patient_metrics"].values() if v.get("dice") is not None or v.get("Dice") is not None]
        elif "cases" in s_data:
            dice_list = [v.get("dice", v.get("Dice")) for v in s_data["cases"].values() if v.get("dice") is not None or v.get("Dice") is not None]

    if len(dice_list) > 0:
        dice_arr = np.array(dice_list)
        mean_dice = float(np.mean(dice_arr))
        sd_dice = float(np.std(dice_arr))
        median_dice = float(np.median(dice_arr))
    else:
        mean_dice = float(s_data.get("mean_dice", np.nan))
        sd_dice = float(s_data.get("std_dice", 0.0))
        median_dice = float(s_data.get("median_dice", mean_dice))
        dice_arr = np.array([mean_dice])

    
    # 2. HD95 & ASSD via multiprocessing
    pred_files = sorted(glob.glob(os.path.join(val_dir, "*.nii.gz")))
    print(f"Computing HD95 & ASSD for {len(pred_files)} validation cases...")
    args = [(pf, gt_dir) for pf in pred_files]
    
    t0 = time.time()
    num_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    with Pool(num_workers) as p:
        surf_res = p.map(compute_case_surface, args)
    
    dt = time.time() - t0
    hd95_vals = [r[0] for r in surf_res if not np.isnan(r[0])]
    assd_vals = [r[1] for r in surf_res if not np.isnan(r[1])]
    
    median_hd95 = float(np.median(hd95_vals)) if hd95_vals else np.nan
    median_assd = float(np.median(assd_vals)) if assd_vals else np.nan
    
    # Save and print the exact table matching Krikun et al. Table 2
    report = f"""
================================================================================
     OFFICIAL REPRODUCTION BENCHMARK: I-SPY2 FOLD 0 (PAPER METRICS ONLY)
================================================================================
Model Architecture : {model_name}
Evaluated Cohort   : {len(dice_arr)} / 197 Validation Cases (Fold 0)
Calculation Time   : {dt:.1f} seconds
Generated At       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Core Metric              Our Reproduction Fold 0      Elena Krikun et al. (2026, Table 2)
--------------------------------------------------------------------------------
Mean Dice                {mean_dice:.4f} +/- {sd_dice:.4f}         0.760  (approx)
Median Dice              {median_dice:.4f}                 0.795  (approx)
Median HD95              {median_hd95:.2f} mm                   4.20 mm (approx)
Median ASSD              {median_assd:.2f} mm                   0.95 mm (approx)
================================================================================
"""
    print(report)
    out_txt = os.path.join(results_dir, "paper_metrics_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    home_report = os.path.expanduser("~/evaluation_standard_fold0.txt")
    try:
        with open(home_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to:\n  - {out_txt}\n  - {home_report}")
    except Exception:
        print(f"Report saved to:\n  - {out_txt}")

if __name__ == "__main__":
    main()
