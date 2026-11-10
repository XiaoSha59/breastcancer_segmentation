#!/usr/bin/env python3
import os
import glob
import json
import time
from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion

val_dir = "/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation"
gt_dir = "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/gt_segmentations"

def compute_case(pf):
    fname = os.path.basename(pf)
    gt_file = os.path.join(gt_dir, fname)
    if not os.path.exists(gt_file):
        return None
    
    try:
        p_img = sitk.ReadImage(pf)
        g_img = sitk.ReadImage(gt_file)
        
        p_arr = sitk.GetArrayFromImage(p_img) > 0
        g_arr = sitk.GetArrayFromImage(g_img) > 0
        spacing = p_img.GetSpacing()[::-1] # Z, Y, X
        
        if not np.any(p_arr) or not np.any(g_arr):
            return {"case": fname, "hd95": np.nan, "assd": np.nan}
        
        pred_border = p_arr ^ binary_erosion(p_arr)
        gt_border = g_arr ^ binary_erosion(g_arr)
        
        dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
        dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
        
        d_pred_to_gt = dt_gt[pred_border]
        d_gt_to_pred = dt_pred[gt_border]
        
        all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        hd95 = float(np.percentile(all_dists, 95))
        assd = float(np.mean(all_dists))
        return {"case": fname, "hd95": hd95, "assd": assd}
    except Exception as e:
        return {"case": fname, "hd95": np.nan, "assd": np.nan}

def main():
    pred_files = sorted(glob.glob(os.path.join(val_dir, "*.nii.gz")))
    print(f"Computing HD95 & ASSD for {len(pred_files)} cases with 8 workers...")
    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(compute_case, pred_files)
    
    dt = time.time() - t0
    results = [r for r in results if r is not None]
    
    hd95_vals = [r["hd95"] for r in results if not np.isnan(r["hd95"])]
    assd_vals = [r["assd"] for r in results if not np.isnan(r["assd"])]
    
    print(f"Done in {dt:.1f}s.")
    print("=== SURFACE METRICS (197 CASES) ===")
    print(f"Cases with valid surface: {len(hd95_vals)} / {len(results)}")
    print(f"HD95 Mean     : {np.mean(hd95_vals):.2f} +/- {np.std(hd95_vals):.2f} mm")
    print(f"HD95 Median   : {np.median(hd95_vals):.2f} mm")
    print(f"HD95 IQR      : [{np.percentile(hd95_vals, 25):.2f}, {np.percentile(hd95_vals, 75):.2f}] mm")
    print(f"ASSD Mean     : {np.mean(assd_vals):.2f} +/- {np.std(assd_vals):.2f} mm")
    print(f"ASSD Median   : {np.median(assd_vals):.2f} mm")
    print(f"ASSD IQR      : [{np.percentile(assd_vals, 25):.2f}, {np.percentile(assd_vals, 75):.2f}] mm")

if __name__ == "__main__":
    main()
