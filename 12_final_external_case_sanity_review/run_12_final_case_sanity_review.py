# run_12_final_case_sanity_review.py
# Final targeted sanity checks for external breast DCE-MRI segmentation analysis.
#
# Purpose:
#   1) Re-select true spatial/boundary mismatch cases using the correct volume-bound signature.
#   2) Quantify LCC-collapse cases over all 526 external cases, including by cohort.
#   3) Check geometry/pairing and possible left/right-side mismatch using centroids.
#   4) Save PNG previews and NIfTI review packets for manual inspection.
#   5) Clean the predictor intercorrelation table by removing trivial self/transform/timing duplicates.
#
# Run from this folder with:
#   python run_12_final_case_sanity_review.py
#
# No command-line parameters are needed. Edit CONFIG only if your folders differ.

from pathlib import Path
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except Exception as e:
    raise RuntimeError("SimpleITK is required. Install with: pip install SimpleITK") from e

try:
    from scipy import ndimage
except Exception as e:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# =============================================================================
# CONFIG — EDIT ONLY HERE IF NEEDED
# =============================================================================
PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# Input tables from previous analyses
CASE_TABLE_10 = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "final_sanity_case_table_with_volume_bounds.csv"
INTERCORR_10 = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "predictor_intercorrelation_pairs.csv"

# Images/masks
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"
PRED_V2_RAW_DIR = Path(r"C:\nnw\mama_mia_output_v2")
INPUT_V2_DIR = Path(r"C:\nnw\mama_mia_input_v2")  # five-channel nnU-Net input, used for previews

# Output folder
OUT_ROOT = PROJECT_ROOT / "12_final_external_case_sanity_review"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
PREVIEW_DIR = OUT_ROOT / "review_packet" / "previews"
NIFTI_DIR = OUT_ROOT / "review_packet" / "nifti_cases"

# Main thresholds for targeted review
RAW_DICE_FOR_LCC_COLLAPSE = 0.20
LCC_DICE_FOR_LCC_COLLAPSE = 0.05
CENTROID_DISTANCE_SUSPECT_MM = 100.0
MIDLINE_MARGIN_MM = 15.0

# If the expected signature column is missing, this fallback defines severe spatial/boundary mismatch
FALLBACK_SPATIAL_DICE_MAX = 0.20
FALLBACK_SPATIAL_THEORETICAL_MIN = 0.70
FALLBACK_SPATIAL_GAP_MIN = 0.50

# Review output limits. Metrics are computed for all cases; previews/NIfTI packets are limited.
MAX_PREVIEW_CASES = 80
MAX_NIFTI_COPY_CASES = 60
DPI = 140

# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dirs():
    for d in [METRICS_DIR, FIGURES_DIR, PREVIEW_DIR, NIFTI_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_case_col(df: pd.DataFrame) -> str:
    for c in ["case_id_std", "case_id", "patient_id", "pid"]:
        if c in df.columns:
            return c
    raise KeyError("Could not find a case ID column. Expected one of: case_id_std, case_id, patient_id, pid")


def find_existing_col(df: pd.DataFrame, candidates, required=True, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns were found: {candidates}")
    return default


def case_id_standard(x) -> str:
    return str(x).replace(".nii.gz", "").replace(".nii", "").strip().upper()


def cohort_from_case(case_id: str) -> str:
    u = case_id.upper()
    if u.startswith("DUKE"):
        return "DUKE"
    if u.startswith("ISPY1"):
        return "ISPY1"
    if u.startswith("NACT"):
        return "NACT"
    return "OTHER"


def find_nifti(folder: Path, case_id: str):
    candidates = [
        folder / f"{case_id}.nii.gz",
        folder / f"{case_id.lower()}.nii.gz",
        folder / f"{case_id.upper()}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_input_channel(case_id: str, channel: str = "0000"):
    candidates = [
        INPUT_V2_DIR / f"{case_id}_{channel}.nii.gz",
        INPUT_V2_DIR / f"{case_id.lower()}_{channel}.nii.gz",
        INPUT_V2_DIR / f"{case_id.upper()}_{channel}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_img(path: Path):
    return sitk.ReadImage(str(path))


def img_to_array(img):
    return sitk.GetArrayFromImage(img)


def binary_array(img):
    return (img_to_array(img) > 0).astype(np.uint8)


def dice_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def precision_recall(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    precision = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    recall = float(tp / gt.sum()) if gt.sum() > 0 else (1.0 if pred.sum() == 0 else 0.0)
    return precision, recall


def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return (lab == largest_label).astype(np.uint8)


def connected_component_features(mask):
    mask = (mask > 0).astype(np.uint8)
    total = int(mask.sum())
    if total == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": 0.0,
            "second_largest_component_fraction": 0.0,
        }
    lab, n = ndimage.label(mask)
    if n == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": 0.0,
            "second_largest_component_fraction": 0.0,
        }
    sizes = np.array(ndimage.sum(mask, lab, index=np.arange(1, n + 1)), dtype=float)
    sizes_sorted = np.sort(sizes)[::-1]
    largest = float(sizes_sorted[0])
    second = float(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0.0
    return {
        "n_components": int(n),
        "largest_component_voxels": int(largest),
        "second_largest_component_voxels": int(second),
        "largest_component_fraction": float(largest / total),
        "second_largest_component_fraction": float(second / total),
    }


def geometry_equal(img_a, img_b, tol_spacing=1e-5, tol_origin=1e-4, tol_direction=1e-5):
    same_size = tuple(img_a.GetSize()) == tuple(img_b.GetSize())
    spacing_diff = float(np.max(np.abs(np.array(img_a.GetSpacing()) - np.array(img_b.GetSpacing()))))
    origin_diff = float(np.linalg.norm(np.array(img_a.GetOrigin()) - np.array(img_b.GetOrigin())))
    direction_diff = float(np.max(np.abs(np.array(img_a.GetDirection()) - np.array(img_b.GetDirection()))))
    return {
        "same_size": bool(same_size),
        "spacing_max_abs_diff": spacing_diff,
        "origin_l2_diff": origin_diff,
        "direction_max_abs_diff": direction_diff,
        "any_geometry_mismatch": bool((not same_size) or spacing_diff > tol_spacing or origin_diff > tol_origin or direction_diff > tol_direction),
    }


def centroid_index_xyz(mask):
    pts_zyx = np.argwhere(mask > 0)
    if len(pts_zyx) == 0:
        return (np.nan, np.nan, np.nan)
    mean_zyx = pts_zyx.mean(axis=0)
    return (float(mean_zyx[2]), float(mean_zyx[1]), float(mean_zyx[0]))


def centroid_physical(img, mask):
    c_xyz = centroid_index_xyz(mask)
    if any(np.isnan(c_xyz)):
        return (np.nan, np.nan, np.nan)
    try:
        return tuple(float(v) for v in img.TransformContinuousIndexToPhysicalPoint(c_xyz))
    except Exception:
        return (np.nan, np.nan, np.nan)


def euclidean_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.nan
    return float(np.linalg.norm(a - b))


def side_label_from_x_index(img, x_idx, margin_mm=MIDLINE_MARGIN_MM):
    if np.isnan(x_idx):
        return "empty"
    size_x = img.GetSize()[0]
    center_x = (size_x - 1) / 2.0
    spacing_x = float(img.GetSpacing()[0]) if img.GetSpacing()[0] != 0 else 1.0
    margin_idx = margin_mm / abs(spacing_x)
    delta = x_idx - center_x
    if abs(delta) <= margin_idx:
        return "midline_or_uncertain"
    return "index_left_half" if delta < 0 else "index_right_half"


def opposite_side(side_a, side_b):
    if "left" in side_a and "right" in side_b:
        return True
    if "right" in side_a and "left" in side_b:
        return True
    return False


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# =============================================================================
# REVIEW COMPUTATION
# =============================================================================

def build_review_table(case_df: pd.DataFrame):
    case_col = find_case_col(case_df)
    raw_dice_col = find_existing_col(case_df, ["v2_raw_dice", "kin_v2_raw_dice"], required=True)
    lcc_dice_col = find_existing_col(case_df, ["v2_lcc_dice", "kin_v2_lcc_dice"], required=True)
    maxdice_col = find_existing_col(
        case_df,
        ["v2_lcc_theoretical_max_dice_by_volume_ratio", "theoretical_max_dice_by_volume_ratio"],
        required=False,
    )
    gap_col = find_existing_col(
        case_df,
        ["v2_lcc_dice_gap_to_volume_bound", "dice_gap_to_volume_bound"],
        required=False,
    )
    sig_col = find_existing_col(
        case_df,
        ["v2_lcc_volume_bound_failure_signature", "volume_bound_failure_signature", "failure_signature"],
        required=False,
    )

    rows = []
    for i, row in case_df.iterrows():
        case_id = case_id_standard(row[case_col])
        if not case_id:
            continue
        if (i + 1) % 50 == 0:
            print(f"  geometry/laterality review {i + 1}/{len(case_df)}")

        gt_path = find_nifti(GT_DIR, case_id.lower())
        pred_path = find_nifti(PRED_V2_RAW_DIR, case_id)

        base = {
            "case_id": case_id,
            "cohort": cohort_from_case(case_id),
            "gt_path": str(gt_path) if gt_path else "",
            "pred_v2_raw_path": str(pred_path) if pred_path else "",
            "raw_dice_table": safe_float(row.get(raw_dice_col, np.nan)),
            "lcc_dice_table": safe_float(row.get(lcc_dice_col, np.nan)),
            "theoretical_max_dice_by_volume_ratio": safe_float(row.get(maxdice_col, np.nan)) if maxdice_col else np.nan,
            "dice_gap_to_volume_bound": safe_float(row.get(gap_col, np.nan)) if gap_col else np.nan,
            "volume_bound_failure_signature": str(row.get(sig_col, "")) if sig_col else "",
            "source_signature_column": sig_col if sig_col else "fallback",
            "missing_gt_or_pred": False,
            "error": "",
        }

        if gt_path is None or pred_path is None:
            base["missing_gt_or_pred"] = True
            rows.append(base)
            continue

        try:
            gt_img = read_img(gt_path)
            pred_img = read_img(pred_path)
            gt = binary_array(gt_img)
            raw = binary_array(pred_img)
            lcc = keep_largest_component(raw)

            geom = geometry_equal(gt_img, pred_img)
            raw_d = dice_score(raw, gt)
            lcc_d = dice_score(lcc, gt)
            raw_p, raw_r = precision_recall(raw, gt)
            lcc_p, lcc_r = precision_recall(lcc, gt)

            gt_c_idx = centroid_index_xyz(gt)
            raw_c_idx = centroid_index_xyz(raw)
            lcc_c_idx = centroid_index_xyz(lcc)
            gt_c_phys = centroid_physical(gt_img, gt)
            raw_c_phys = centroid_physical(gt_img, raw)
            lcc_c_phys = centroid_physical(gt_img, lcc)

            gt_side = side_label_from_x_index(gt_img, gt_c_idx[0])
            raw_side = side_label_from_x_index(gt_img, raw_c_idx[0])
            lcc_side = side_label_from_x_index(gt_img, lcc_c_idx[0])

            raw_comp = connected_component_features(raw)
            lcc_comp = connected_component_features(lcc)

            gt_vox = int(gt.sum())
            raw_vox = int(raw.sum())
            lcc_vox = int(lcc.sum())
            voxel_volume_mm3 = float(np.prod(gt_img.GetSpacing()))
            gt_vol_cm3 = gt_vox * voxel_volume_mm3 / 1000.0
            raw_vol_cm3 = raw_vox * voxel_volume_mm3 / 1000.0
            lcc_vol_cm3 = lcc_vox * voxel_volume_mm3 / 1000.0

            raw_to_lcc_delta = lcc_d - raw_d
            lcc_collapse = bool(raw_d > RAW_DICE_FOR_LCC_COLLAPSE and lcc_d < LCC_DICE_FOR_LCC_COLLAPSE)
            lcc_wrong_side = opposite_side(gt_side, lcc_side)
            raw_wrong_side = opposite_side(gt_side, raw_side)
            lcc_dist = euclidean_distance(gt_c_phys, lcc_c_phys)
            raw_dist = euclidean_distance(gt_c_phys, raw_c_phys)

            fallback_spatial = bool(
                lcc_d <= FALLBACK_SPATIAL_DICE_MAX
                and (np.isnan(base["theoretical_max_dice_by_volume_ratio"]) or base["theoretical_max_dice_by_volume_ratio"] >= FALLBACK_SPATIAL_THEORETICAL_MIN)
                and (np.isnan(base["dice_gap_to_volume_bound"]) or base["dice_gap_to_volume_bound"] >= FALLBACK_SPATIAL_GAP_MIN)
            )
            explicit_spatial = str(base["volume_bound_failure_signature"]) == "spatial_or_boundary_mismatch_beyond_volume"
            spatial_signature = bool(explicit_spatial or (sig_col is None and fallback_spatial))

            base.update(geom)
            base.update({
                "raw_dice_recomputed": raw_d,
                "lcc_dice_recomputed": lcc_d,
                "raw_precision_recomputed": raw_p,
                "raw_recall_recomputed": raw_r,
                "lcc_precision_recomputed": lcc_p,
                "lcc_recall_recomputed": lcc_r,
                "raw_to_lcc_delta_dice": raw_to_lcc_delta,
                "table_raw_dice_matches_recomputed": bool(np.isclose(base["raw_dice_table"], raw_d, atol=1e-4, equal_nan=True)),
                "table_lcc_dice_matches_recomputed": bool(np.isclose(base["lcc_dice_table"], lcc_d, atol=1e-4, equal_nan=True)),
                "gt_voxels": gt_vox,
                "raw_voxels": raw_vox,
                "lcc_voxels": lcc_vox,
                "gt_vol_cm3": gt_vol_cm3,
                "raw_vol_cm3": raw_vol_cm3,
                "lcc_vol_cm3": lcc_vol_cm3,
                "raw_pred_to_gt_vol_ratio_recomputed": raw_vol_cm3 / gt_vol_cm3 if gt_vol_cm3 > 0 else np.nan,
                "lcc_pred_to_gt_vol_ratio_recomputed": lcc_vol_cm3 / gt_vol_cm3 if gt_vol_cm3 > 0 else np.nan,
                "gt_centroid_x_index": gt_c_idx[0],
                "gt_centroid_y_index": gt_c_idx[1],
                "gt_centroid_z_index": gt_c_idx[2],
                "raw_centroid_x_index": raw_c_idx[0],
                "raw_centroid_y_index": raw_c_idx[1],
                "raw_centroid_z_index": raw_c_idx[2],
                "lcc_centroid_x_index": lcc_c_idx[0],
                "lcc_centroid_y_index": lcc_c_idx[1],
                "lcc_centroid_z_index": lcc_c_idx[2],
                "gt_centroid_x_phys": gt_c_phys[0],
                "gt_centroid_y_phys": gt_c_phys[1],
                "gt_centroid_z_phys": gt_c_phys[2],
                "raw_centroid_x_phys": raw_c_phys[0],
                "raw_centroid_y_phys": raw_c_phys[1],
                "raw_centroid_z_phys": raw_c_phys[2],
                "lcc_centroid_x_phys": lcc_c_phys[0],
                "lcc_centroid_y_phys": lcc_c_phys[1],
                "lcc_centroid_z_phys": lcc_c_phys[2],
                "raw_centroid_distance_to_gt_mm": raw_dist,
                "lcc_centroid_distance_to_gt_mm": lcc_dist,
                "raw_centroid_distance_gt_100mm": bool(raw_dist > CENTROID_DISTANCE_SUSPECT_MM) if not np.isnan(raw_dist) else False,
                "lcc_centroid_distance_gt_100mm": bool(lcc_dist > CENTROID_DISTANCE_SUSPECT_MM) if not np.isnan(lcc_dist) else False,
                "gt_index_side": gt_side,
                "raw_index_side": raw_side,
                "lcc_index_side": lcc_side,
                "raw_opposite_side_suspect": bool(raw_wrong_side),
                "lcc_opposite_side_suspect": bool(lcc_wrong_side),
                "lcc_collapse_raw_gt_0p2_lcc_lt_0p05": lcc_collapse,
                "spatial_boundary_signature": spatial_signature,
                "explicit_spatial_boundary_signature": explicit_spatial,
                "fallback_spatial_boundary_signature": fallback_spatial,
            })
            # Component features with prefixes
            for k, v in raw_comp.items():
                base[f"raw_{k}"] = v
            for k, v in lcc_comp.items():
                base[f"lcc_{k}"] = v

            # A compact review flag
            base["needs_manual_review"] = bool(
                base["any_geometry_mismatch"]
                or lcc_collapse
                or spatial_signature
                or base["lcc_centroid_distance_gt_100mm"]
                or base["lcc_opposite_side_suspect"]
            )
        except Exception as e:
            base["error"] = repr(e)
        rows.append(base)

    review = pd.DataFrame(rows)
    return review

# =============================================================================
# OUTPUT TABLES AND FIGURES
# =============================================================================

def summarize_boolean_by_cohort(df, flag_col, out_name):
    rows = []
    for cohort, grp in df.groupby("cohort", dropna=False):
        denom = len(grp)
        count = int(grp[flag_col].fillna(False).sum()) if flag_col in grp.columns else 0
        rows.append({"cohort": cohort, "n": denom, flag_col: count, "rate": count / denom if denom else np.nan})
    out = pd.DataFrame(rows).sort_values("cohort")
    out.to_csv(METRICS_DIR / out_name, index=False)
    return out


def write_summary_tables(review):
    review.to_csv(METRICS_DIR / "full_geometry_laterality_lcc_review_all_cases.csv", index=False)

    # Corrected spatial signature cases
    spatial = review[review["spatial_boundary_signature"].fillna(False)].copy()
    spatial.sort_values(["lcc_dice_recomputed", "dice_gap_to_volume_bound"], ascending=[True, False], inplace=True)
    spatial.to_csv(METRICS_DIR / "corrected_spatial_boundary_mismatch_cases.csv", index=False)

    # LCC collapse cases over all 526
    collapse = review[review["lcc_collapse_raw_gt_0p2_lcc_lt_0p05"].fillna(False)].copy()
    collapse.sort_values(["raw_to_lcc_delta_dice", "raw_dice_recomputed"], ascending=[True, False], inplace=True)
    collapse.to_csv(METRICS_DIR / "lcc_collapse_cases_all_external.csv", index=False)

    # Side / laterality suspect cases
    side = review[review["lcc_opposite_side_suspect"].fillna(False) | review["raw_opposite_side_suspect"].fillna(False)].copy()
    side.sort_values(["lcc_centroid_distance_to_gt_mm"], ascending=False, inplace=True)
    side.to_csv(METRICS_DIR / "opposite_side_suspect_cases.csv", index=False)

    # Large displacement cases
    large_dist = review[review["lcc_centroid_distance_gt_100mm"].fillna(False)].copy()
    large_dist.sort_values(["lcc_centroid_distance_to_gt_mm"], ascending=False, inplace=True)
    large_dist.to_csv(METRICS_DIR / "large_centroid_distance_gt_100mm_cases.csv", index=False)

    # Summaries
    summarize_boolean_by_cohort(review, "lcc_collapse_raw_gt_0p2_lcc_lt_0p05", "lcc_collapse_summary_by_cohort.csv")
    summarize_boolean_by_cohort(review, "spatial_boundary_signature", "spatial_boundary_signature_summary_by_cohort.csv")
    summarize_boolean_by_cohort(review, "lcc_opposite_side_suspect", "lcc_opposite_side_summary_by_cohort.csv")
    summarize_boolean_by_cohort(review, "lcc_centroid_distance_gt_100mm", "large_centroid_distance_summary_by_cohort.csv")

    # LCC trade-off summary
    trade_rows = []
    for cohort, grp in review.groupby("cohort", dropna=False):
        trade_rows.append({
            "cohort": cohort,
            "n": len(grp),
            "raw_dice_mean": grp["raw_dice_recomputed"].mean(),
            "lcc_dice_mean": grp["lcc_dice_recomputed"].mean(),
            "mean_delta_lcc_minus_raw": grp["raw_to_lcc_delta_dice"].mean(),
            "lcc_helped_delta_gt_0p05_n": int((grp["raw_to_lcc_delta_dice"] > 0.05).sum()),
            "lcc_hurt_delta_lt_minus_0p05_n": int((grp["raw_to_lcc_delta_dice"] < -0.05).sum()),
            "lcc_collapse_n": int(grp["lcc_collapse_raw_gt_0p2_lcc_lt_0p05"].sum()),
        })
    trade = pd.DataFrame(trade_rows)
    trade.to_csv(METRICS_DIR / "lcc_tradeoff_summary_by_cohort.csv", index=False)

    return spatial, collapse, side, large_dist, trade


def create_preview(case_id, review_row, out_path):
    gt_path = Path(review_row["gt_path"])
    pred_path = Path(review_row["pred_v2_raw_path"])
    img_path = find_input_channel(case_id, "0000") or find_input_channel(case_id, "0001")

    gt_img = read_img(gt_path)
    pred_img = read_img(pred_path)
    gt = binary_array(gt_img)
    raw = binary_array(pred_img)
    lcc = keep_largest_component(raw)

    if img_path is not None:
        try:
            bg = img_to_array(read_img(img_path)).astype(float)
        except Exception:
            bg = np.zeros_like(gt, dtype=float)
    else:
        bg = np.zeros_like(gt, dtype=float)

    # Pick slice: prefer GT centroid, otherwise raw/LCC centroid
    if gt.sum() > 0:
        z = int(round(centroid_index_xyz(gt)[2]))
    elif lcc.sum() > 0:
        z = int(round(centroid_index_xyz(lcc)[2]))
    elif raw.sum() > 0:
        z = int(round(centroid_index_xyz(raw)[2]))
    else:
        z = bg.shape[0] // 2
    z = max(0, min(z, bg.shape[0] - 1))

    bg_slice = bg[z]
    gt_slice = gt[z]
    raw_slice = raw[z]
    lcc_slice = lcc[z]

    # Robust display window
    vals = bg_slice[np.isfinite(bg_slice)]
    if vals.size > 0:
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max() + 1e-6)
    else:
        lo, hi = 0, 1

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=DPI)
    titles = ["Input image", "GT + raw", "GT + LCC", "Raw vs LCC"]
    for ax, title in zip(axes, titles):
        ax.imshow(bg_slice, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Contours: GT green, raw red, LCC yellow/blue
    for ax in [axes[1], axes[2]]:
        if gt_slice.sum() > 0:
            ax.contour(gt_slice, levels=[0.5], colors="lime", linewidths=1.4)
    if raw_slice.sum() > 0:
        axes[1].contour(raw_slice, levels=[0.5], colors="red", linewidths=1.2)
        axes[3].contour(raw_slice, levels=[0.5], colors="red", linewidths=1.2)
    if lcc_slice.sum() > 0:
        axes[2].contour(lcc_slice, levels=[0.5], colors="yellow", linewidths=1.4)
        axes[3].contour(lcc_slice, levels=[0.5], colors="cyan", linewidths=1.2)

    fig.suptitle(
        f"{case_id} | cohort={review_row.get('cohort','')} | "
        f"raw Dice={review_row.get('raw_dice_recomputed', np.nan):.3f} | "
        f"LCC Dice={review_row.get('lcc_dice_recomputed', np.nan):.3f} | "
        f"dist={review_row.get('lcc_centroid_distance_to_gt_mm', np.nan):.1f}mm | "
        f"GT side={review_row.get('gt_index_side','')} / LCC side={review_row.get('lcc_index_side','')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def copy_nifti_packet(case_id, review_row):
    case_dir = NIFTI_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    gt_path = Path(review_row["gt_path"])
    pred_path = Path(review_row["pred_v2_raw_path"])
    img_path = find_input_channel(case_id, "0000") or find_input_channel(case_id, "0001")

    # Copy raw files
    if gt_path.exists():
        shutil.copy2(gt_path, case_dir / f"{case_id}_gt.nii.gz")
    if pred_path.exists():
        shutil.copy2(pred_path, case_dir / f"{case_id}_pred_v2_raw.nii.gz")
    if img_path is not None and img_path.exists():
        shutil.copy2(img_path, case_dir / f"{case_id}_input_reference.nii.gz")

    # Save recomputed LCC as NIfTI
    try:
        pred_img = read_img(pred_path)
        lcc = keep_largest_component(binary_array(pred_img))
        lcc_img = sitk.GetImageFromArray(lcc.astype(np.uint8))
        lcc_img.CopyInformation(pred_img)
        sitk.WriteImage(lcc_img, str(case_dir / f"{case_id}_pred_v2_lcc_recomputed.nii.gz"), useCompression=True)
    except Exception:
        pass


def create_review_packets(review):
    # Select cases for preview/NIfTI with reasons.
    reasons = []
    for _, r in review.iterrows():
        tags = []
        if bool(r.get("spatial_boundary_signature", False)):
            tags.append("spatial_boundary_signature")
        if bool(r.get("lcc_collapse_raw_gt_0p2_lcc_lt_0p05", False)):
            tags.append("lcc_collapse")
        if bool(r.get("lcc_opposite_side_suspect", False)):
            tags.append("opposite_side_suspect")
        if bool(r.get("lcc_centroid_distance_gt_100mm", False)):
            tags.append("large_centroid_distance")
        if tags:
            rr = r.to_dict()
            rr["review_tags"] = ";".join(tags)
            reasons.append(rr)
    selected = pd.DataFrame(reasons)
    if selected.empty:
        selected = review.sort_values("dice_gap_to_volume_bound", ascending=False).head(30).copy()
        selected["review_tags"] = "top_gap_fallback"
    else:
        # Prioritize true spatial signature and LCC collapse, then largest centroid distance and gap.
        selected["priority"] = (
            selected.get("spatial_boundary_signature", False).astype(int) * 100
            + selected.get("lcc_collapse_raw_gt_0p2_lcc_lt_0p05", False).astype(int) * 50
            + selected.get("lcc_opposite_side_suspect", False).astype(int) * 25
            + selected.get("lcc_centroid_distance_gt_100mm", False).astype(int) * 10
        )
        selected.sort_values(["priority", "lcc_centroid_distance_to_gt_mm", "dice_gap_to_volume_bound"], ascending=[False, False, False], inplace=True)

    selected_manifest = selected.copy()
    selected_manifest.to_csv(METRICS_DIR / "selected_manual_review_manifest.csv", index=False)

    # Manual checklist file the user can edit later.
    checklist_cols = [
        "case_id", "cohort", "review_tags", "raw_dice_recomputed", "lcc_dice_recomputed",
        "gt_index_side", "raw_index_side", "lcc_index_side", "raw_opposite_side_suspect", "lcc_opposite_side_suspect",
        "raw_centroid_distance_to_gt_mm", "lcc_centroid_distance_to_gt_mm",
        "manual_visual_verdict", "notes",
    ]
    checklist = selected_manifest.copy()
    checklist["manual_visual_verdict"] = ""
    checklist["notes"] = ""
    for c in checklist_cols:
        if c not in checklist.columns:
            checklist[c] = ""
    checklist[checklist_cols].to_csv(METRICS_DIR / "manual_review_checklist.csv", index=False)

    # Create previews and NIfTI packets.
    preview_cases = selected_manifest.head(MAX_PREVIEW_CASES)
    png_paths = []
    for _, r in preview_cases.iterrows():
        case_id = r["case_id"]
        out_png = PREVIEW_DIR / f"{case_id}_review.png"
        try:
            create_preview(case_id, r, out_png)
            png_paths.append(out_png)
        except Exception as e:
            print(f"  preview failed for {case_id}: {e}")

    for _, r in selected_manifest.head(MAX_NIFTI_COPY_CASES).iterrows():
        try:
            copy_nifti_packet(r["case_id"], r)
        except Exception as e:
            print(f"  NIfTI packet failed for {r['case_id']}: {e}")

    # Contact sheet
    if PIL_AVAILABLE and png_paths:
        try:
            make_contact_sheet(png_paths, PREVIEW_DIR / "contact_sheet_selected_review_cases.jpg")
        except Exception as e:
            print(f"  contact sheet failed: {e}")

    return selected_manifest


def make_contact_sheet(image_paths, out_path, thumb_w=520):
    imgs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = thumb_w / float(w)
        im = im.resize((thumb_w, max(1, int(h * scale))))
        imgs.append(im)
    if not imgs:
        return
    cols = 2
    rows = int(np.ceil(len(imgs) / cols))
    cell_w = max(im.size[0] for im in imgs)
    cell_h = max(im.size[1] for im in imgs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, im in enumerate(imgs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
    sheet.save(out_path, quality=90)

# =============================================================================
# CLEAN INTERCORRELATIONS
# =============================================================================

def feature_family(name: str) -> str:
    s = str(name).lower()
    if any(k in s for k in ["dice", "precision", "recall", "iou", "hd95", "assd"]):
        return "performance"
    if any(k in s for k in ["n_components", "largest_component", "second_largest", "component_fraction", "component"]):
        return "component_structure"
    if any(k in s for k in ["pred_to_gt_vol", "vol_ratio", "volume_bound", "theoretical", "gap_to_volume", "pred_vol", "gt_vol", "log_gt_vol", "voxels", "vol_cm3"]):
        return "volume"
    if any(k in s for k in ["first_post", "last_post", "early_dev", "late_dev", "time", "timing"]):
        return "timing"
    if any(k in s for k in ["enh", "auc", "washout", "slope", "ring", "fg_", "tumor_to", "kin_"]):
        return "contrast_kinetic"
    if any(k in s for k in ["cohort", "phase", "quality", "dataset"]):
        return "cohort_protocol"
    return "other"


def canonical_feature(name: str) -> str:
    s = str(name).lower()
    # Remove common prefixes indicating version/source, not feature meaning.
    for pref in ["kin_", "v1_", "v2_", "raw_", "lcc_", "pred_", "gt_", "fg_", "ring_"]:
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.replace("log_", "")
    s = s.replace("_recomputed", "")
    # Collapse ROI-specific timing duplicates.
    for t in ["first_post_s", "last_post_s", "early_dev_s", "late_dev_s"]:
        if t in s:
            return t
    # Collapse common volume variants only when same core variable.
    return s


def is_trivial_pair(a, b):
    a = str(a).lower()
    b = str(b).lower()
    ca = canonical_feature(a)
    cb = canonical_feature(b)
    fa = feature_family(a)
    fb = feature_family(b)

    # Exact canonical duplicate or log/self transform.
    if ca == cb:
        return True
    if ca.replace("log_", "") == cb.replace("log_", ""):
        return True

    # Timing-vs-timing pairs are mostly duplicates/derived variables in this project.
    if fa == "timing" and fb == "timing":
        return True

    # ROI-specific kinetic AUC/max pairs can be very close, but keep them if across different families.
    # Remove only obvious same-ROI max/AUC duplicates with abs rho near 1 later via canonical-like names.
    if fa == fb == "contrast_kinetic":
        # Same ROI and one max vs auc is often redundant descriptive summary.
        roi_tokens = ["gt", "fg", "ring", "tumor_to_ring", "tumor_to_fg"]
        same_roi = any((rt in a and rt in b) for rt in roi_tokens)
        if same_roi and (("max_enh" in a and "auc_enh" in b) or ("auc_enh" in a and "max_enh" in b)):
            return True

    return False


def clean_intercorrelation_table():
    if not INTERCORR_10.exists():
        print(f"Intercorrelation table not found: {INTERCORR_10}")
        return pd.DataFrame(), pd.DataFrame()
    corr = pd.read_csv(INTERCORR_10)
    if "feature_a" not in corr.columns or "feature_b" not in corr.columns:
        print("Intercorrelation table missing feature_a/feature_b columns")
        return pd.DataFrame(), pd.DataFrame()

    corr["family_a"] = corr["feature_a"].apply(feature_family)
    corr["family_b"] = corr["feature_b"].apply(feature_family)
    corr["canonical_a"] = corr["feature_a"].apply(canonical_feature)
    corr["canonical_b"] = corr["feature_b"].apply(canonical_feature)
    corr["is_trivial_or_self_transform_pair"] = [is_trivial_pair(a, b) for a, b in zip(corr["feature_a"], corr["feature_b"])]

    rho_col = "spearman_rho" if "spearman_rho" in corr.columns else None
    if rho_col:
        corr["abs_rho"] = corr[rho_col].abs()
    else:
        corr["abs_rho"] = np.nan

    nontrivial = corr[~corr["is_trivial_or_self_transform_pair"]].copy()
    nontrivial.sort_values("abs_rho", ascending=False, inplace=True)
    nontrivial.to_csv(METRICS_DIR / "cleaned_intercorrelations_no_self_or_timing_duplicates.csv", index=False)

    # Cross-family only table — most useful for supplement.
    cross_family = nontrivial[nontrivial["family_a"] != nontrivial["family_b"]].copy()
    cross_family.sort_values("abs_rho", ascending=False, inplace=True)
    cross_family.to_csv(METRICS_DIR / "cleaned_intercorrelations_cross_family_only.csv", index=False)

    return nontrivial, cross_family

# =============================================================================
# FIGURES
# =============================================================================

def fig_lcc_collapse_by_cohort(review):
    summ = summarize_boolean_by_cohort(review, "lcc_collapse_raw_gt_0p2_lcc_lt_0p05", "_tmp_lcc_collapse_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    ax.bar(summ["cohort"], summ["rate"])
    ax.set_ylabel("Rate of LCC-collapse cases")
    ax.set_title("LCC-collapse frequency by cohort")
    ax.set_ylim(0, max(0.05, float(summ["rate"].max()) * 1.25))
    for i, r in summ.iterrows():
        ax.text(i, r["rate"], f"{int(r['lcc_collapse_raw_gt_0p2_lcc_lt_0p05'])}/{int(r['n'])}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_lcc_collapse_rate_by_cohort.png", bbox_inches="tight")
    plt.close(fig)


def fig_raw_vs_lcc_dice(review):
    fig, ax = plt.subplots(figsize=(5.6, 5), dpi=DPI)
    for cohort, grp in review.groupby("cohort"):
        ax.scatter(grp["raw_dice_recomputed"], grp["lcc_dice_recomputed"], s=14, alpha=0.55, label=cohort)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.axhline(LCC_DICE_FOR_LCC_COLLAPSE, linestyle=":", linewidth=1)
    ax.axvline(RAW_DICE_FOR_LCC_COLLAPSE, linestyle=":", linewidth=1)
    ax.set_xlabel("v2 raw Dice")
    ax.set_ylabel("v2 LCC Dice")
    ax.set_title("Case-level LCC trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_raw_vs_lcc_dice_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_centroid_distance_by_group(review):
    # Use selected key groups, not all, to avoid unreadable plot.
    tmp = review.copy()
    tmp["plot_group"] = "other"
    tmp.loc[tmp["spatial_boundary_signature"].fillna(False), "plot_group"] = "spatial signature"
    tmp.loc[tmp["lcc_collapse_raw_gt_0p2_lcc_lt_0p05"].fillna(False), "plot_group"] = "LCC collapse"
    tmp.loc[tmp["lcc_opposite_side_suspect"].fillna(False), "plot_group"] = "opposite side"
    order = ["other", "spatial signature", "LCC collapse", "opposite side"]
    data = [tmp.loc[tmp["plot_group"] == g, "lcc_centroid_distance_to_gt_mm"].dropna().values for g in order]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("GT-to-LCC centroid distance (mm)")
    ax.set_title("Large spatial displacement in severe-failure groups")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_centroid_distance_by_review_group.png", bbox_inches="tight")
    plt.close(fig)


def fig_opposite_side_counts(review):
    flags = []
    for cohort, grp in review.groupby("cohort"):
        flags.append({
            "cohort": cohort,
            "raw_opposite_side": int(grp["raw_opposite_side_suspect"].sum()),
            "lcc_opposite_side": int(grp["lcc_opposite_side_suspect"].sum()),
            "n": len(grp),
        })
    df = pd.DataFrame(flags)
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
    ax.bar(x - width / 2, df["raw_opposite_side"], width, label="raw")
    ax.bar(x + width / 2, df["lcc_opposite_side"], width, label="LCC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cohort"])
    ax.set_ylabel("Count")
    ax.set_title("Opposite-side centroid flags by cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_opposite_side_flags_by_cohort.png", bbox_inches="tight")
    plt.close(fig)


def create_figures(review):
    fig_lcc_collapse_by_cohort(review)
    fig_raw_vs_lcc_dice(review)
    fig_centroid_distance_by_group(review)
    fig_opposite_side_counts(review)

# =============================================================================
# TAKEAWAYS
# =============================================================================

def write_takeaways(review, selected, nontrivial_corr, cross_family_corr):
    n = len(review)
    spatial_n = int(review["spatial_boundary_signature"].sum())
    collapse_n = int(review["lcc_collapse_raw_gt_0p2_lcc_lt_0p05"].sum())
    geom_mismatch_n = int(review["any_geometry_mismatch"].sum())
    lcc_opp_n = int(review["lcc_opposite_side_suspect"].sum())
    raw_opp_n = int(review["raw_opposite_side_suspect"].sum())
    large_dist_n = int(review["lcc_centroid_distance_gt_100mm"].sum())

    lines = []
    lines.append("12 FINAL EXTERNAL CASE SANITY REVIEW — QUICK TAKEAWAYS")
    lines.append("=" * 76)
    lines.append(f"Cases reviewed quantitatively: {n}")
    lines.append("")
    lines.append("Geometry and pairing checks:")
    lines.append(f"  Any geometry mismatch: {geom_mismatch_n}/{n}")
    lines.append(f"  Table raw Dice matched recomputation: {int(review['table_raw_dice_matches_recomputed'].sum())}/{n}")
    lines.append(f"  Table LCC Dice matched recomputation: {int(review['table_lcc_dice_matches_recomputed'].sum())}/{n}")
    lines.append("")
    lines.append("Corrected severe-failure selection:")
    lines.append(f"  Spatial/boundary signature cases: {spatial_n}/{n}")
    lines.append(f"  Large LCC centroid distance > {CENTROID_DISTANCE_SUSPECT_MM:.0f} mm: {large_dist_n}/{n}")
    lines.append(f"  Raw opposite-side flags: {raw_opp_n}/{n}")
    lines.append(f"  LCC opposite-side flags: {lcc_opp_n}/{n}")
    lines.append("")
    lines.append("LCC trade-off:")
    lines.append(f"  LCC-collapse cases (raw Dice > {RAW_DICE_FOR_LCC_COLLAPSE}, LCC Dice < {LCC_DICE_FOR_LCC_COLLAPSE}): {collapse_n}/{n}")
    if n:
        lines.append(f"  LCC-collapse rate: {collapse_n / n:.3%}")
    lines.append("")
    lines.append("LCC-collapse by cohort:")
    for cohort, grp in review.groupby("cohort"):
        c = int(grp["lcc_collapse_raw_gt_0p2_lcc_lt_0p05"].sum())
        lines.append(f"  {cohort}: {c}/{len(grp)} ({c / len(grp):.2%})")
    lines.append("")
    lines.append("Opposite-side LCC flags by cohort:")
    for cohort, grp in review.groupby("cohort"):
        c = int(grp["lcc_opposite_side_suspect"].sum())
        lines.append(f"  {cohort}: {c}/{len(grp)} ({c / len(grp):.2%})")
    lines.append("")
    lines.append("Cleaned intercorrelations:")
    if not cross_family_corr.empty:
        top = cross_family_corr.head(8)
        for _, r in top.iterrows():
            lines.append(f"  {r['feature_a']} vs {r['feature_b']}: rho={r.get('spearman_rho', np.nan):.3f}, families={r['family_a']} vs {r['family_b']}")
    else:
        lines.append("  No cleaned cross-family correlations available.")
    lines.append("")
    lines.append("Manuscript-safe framing:")
    lines.append("  Use this as a targeted sanity review, not as proof that every severe failure was manually audited.")
    lines.append("  If geometry mismatches are zero and recomputed Dice matches the tables, external failures are unlikely to be caused by a global spacing/orientation error.")
    lines.append("  Opposite-side flags and large centroid distances should be described as suspect spatial-displacement patterns requiring qualitative review, not automatically as patient/file mismatch.")
    lines.append("  LCC should be framed as average-improving but case-risky postprocessing, because it can collapse partially correct raw predictions when the largest component is not tumor-overlapping.")

    (OUT_ROOT / "article_takeaways_12.txt").write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("12 Final external case sanity review")
    print("=" * 42)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUT_ROOT}")
    print(f"Case table   : {CASE_TABLE_10}")
    ensure_dirs()

    if not CASE_TABLE_10.exists():
        raise FileNotFoundError(f"Missing case table: {CASE_TABLE_10}")
    case_df = pd.read_csv(CASE_TABLE_10)
    print(f"Cases loaded : {len(case_df)}")

    print("\nComputing full geometry/laterality/LCC review over all cases...")
    review = build_review_table(case_df)

    print("\nWriting summary tables...")
    spatial, collapse, side, large_dist, trade = write_summary_tables(review)

    print("\nCreating selected review packet...")
    selected = create_review_packets(review)

    print("\nCleaning intercorrelation table...")
    nontrivial_corr, cross_family_corr = clean_intercorrelation_table()

    print("\nCreating figures...")
    create_figures(review)

    write_takeaways(review, selected, nontrivial_corr, cross_family_corr)

    print("\nDONE — 12 final external case sanity review complete.")
    print(f"Metrics      : {METRICS_DIR}")
    print(f"Figures      : {FIGURES_DIR}")
    print(f"Review packet: {OUT_ROOT / 'review_packet'}")
    print(f"Takeaways    : {OUT_ROOT / 'article_takeaways_12.txt'}")
    print("\nQuick results:")
    print(f"  all cases reviewed: {len(review)}")
    print(f"  geometry mismatch cases: {int(review['any_geometry_mismatch'].sum())}")
    print(f"  spatial/boundary signature cases: {int(review['spatial_boundary_signature'].sum())}")
    print(f"  LCC-collapse cases: {int(review['lcc_collapse_raw_gt_0p2_lcc_lt_0p05'].sum())}")
    print(f"  LCC opposite-side flags: {int(review['lcc_opposite_side_suspect'].sum())}")
    print(f"  large LCC centroid distance > {CENTROID_DISTANCE_SUSPECT_MM:.0f}mm: {int(review['lcc_centroid_distance_gt_100mm'].sum())}")
    if not cross_family_corr.empty:
        print("\nTop cleaned cross-family correlations:")
        cols = ["feature_a", "feature_b", "family_a", "family_b", "spearman_rho"]
        print(cross_family_corr[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        print("If full review CSV was already saved, send me the last lines and I will make a finish-from-saved-CSV helper.")
        raise
