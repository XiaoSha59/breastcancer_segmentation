#!/usr/bin/env python3
import os
import glob
import json
import time
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion

val_dir = "/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation"
gt_dir = "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/gt_segmentations"

pred_files = sorted(glob.glob(os.path.join(val_dir, "*.nii.gz")))
print(f"Found {len(pred_files)} predicted cases.")

def compute_surface_distances(pred_arr, gt_arr, spacing):
    if not np.any(pred_arr) or not np.any(gt_arr):
        return np.nan, np.nan
    
    # Extract surface boundaries
    pred_border = pred_arr ^ binary_erosion(pred_arr)
    gt_border = gt_arr ^ binary_erosion(gt_arr)
    
    # Distance transforms with voxel spacing in mm
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    
    # Surface distances
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]
    
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = np.percentile(all_dists, 95)
    assd = np.mean(all_dists)
    return hd95, assd

# Test on first 10 cases to estimate time
hd95_list = []
assd_list = []

t0 = time.time()
sample_n = min(20, len(pred_files))
for i, pf in enumerate(pred_files[:sample_n]):
    fname = os.path.basename(pf)
    gt_file = os.path.join(gt_dir, fname)
    if not os.path.exists(gt_file):
        continue
    
    p_img = sitk.ReadImage(pf)
    g_img = sitk.ReadImage(gt_file)
    
    p_arr = sitk.GetArrayFromImage(p_img) > 0
    g_arr = sitk.GetArrayFromImage(g_img) > 0
    spacing = p_img.GetSpacing()[::-1] # Z, Y, X in numpy order
    
    hd95, assd = compute_surface_distances(p_arr, g_arr, spacing)
    if not np.isnan(hd95):
        hd95_list.append(hd95)
        assd_list.append(assd)

dt = time.time() - t0
print(f"Processed {sample_n} cases in {dt:.1f}s (~{dt/sample_n:.2f}s/case)")
if hd95_list:
    print(f"Sample ({sample_n} cases) Median HD95: {np.median(hd95_list):.2f} mm")
    print(f"Sample ({sample_n} cases) Median ASSD: {np.median(assd_list):.2f} mm")
