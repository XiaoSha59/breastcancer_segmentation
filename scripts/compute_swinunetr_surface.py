import os
import glob
from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion

val_dir = "/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_200epochs/fold_0/validation"
gt_dir = "/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/labelsTr"

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
        if p_arr.shape != g_arr.shape:
            # Handle SimpleITK / PyTorch axis transpose (X, Y, Z) vs (Z, Y, X)
            for perm in [(2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1)]:
                if np.transpose(p_arr, perm).shape == g_arr.shape:
                    p_arr = np.transpose(p_arr, perm)
                    break
        spacing = p_img.GetSpacing()[::-1]
        if not np.any(p_arr) or not np.any(g_arr):
            return None
        pred_b = p_arr ^ binary_erosion(p_arr)
        gt_b = g_arr ^ binary_erosion(g_arr)
        dt_pred = distance_transform_edt(~pred_b, sampling=spacing)
        dt_gt = distance_transform_edt(~gt_b, sampling=spacing)
        d_pred_to_gt = dt_gt[pred_b]
        d_gt_to_pred = dt_pred[gt_b]
        all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
        return float(np.percentile(all_d, 95)), float(np.mean(all_d))
    except Exception as e:
        print(f"Error {fname}: {type(e).__name__}: {e}")
        return None

if __name__ == "__main__":
    pfs = sorted(glob.glob(os.path.join(val_dir, "*.nii.gz")))
    with Pool(8) as p:
        res = p.map(compute_case, pfs)
    valid_res = [r for r in res if r is not None]
    hd95s = [r[0] for r in valid_res if not np.isnan(r[0])]
    assds = [r[1] for r in valid_res if not np.isnan(r[1])]
    print(f"Valid surface cases: {len(hd95s)} / {len(pfs)}")
    print(f"Median HD95: {np.median(hd95s):.2f} mm")
    print(f"Mean HD95:   {np.mean(hd95s):.2f} mm")
    print(f"Median ASSD: {np.median(assds):.2f} mm")
    print(f"Mean ASSD:   {np.mean(assds):.2f} mm")
