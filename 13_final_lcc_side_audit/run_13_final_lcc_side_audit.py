# run_13_final_lcc_side_audit.py
# Final audit for LCC-collapse, opposite-half flags, cohort concentration, and cleaned collinearity.
# Run from: C:\Users\olegk\Desktop\MRI Project\13_final_lcc_side_audit
# Command: python run_13_final_lcc_side_audit.py

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except Exception as e:
    raise SystemExit("ERROR: SimpleITK is required. Install with: pip install SimpleITK") from e

try:
    from scipy import ndimage, stats
except Exception as e:
    raise SystemExit("ERROR: scipy is required. Install with: pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# CONFIG — edit only here if your paths differ
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")
OUTPUT_ROOT = PROJECT_ROOT / "13_final_lcc_side_audit"

CASE_TABLE = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "final_sanity_case_table_with_volume_bounds.csv"
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"
IMAGES_DIR = PROJECT_ROOT / "images_2"

# Optional existing 12 output, used only for comparison with previous flags.
EXISTING_12_REVIEW = PROJECT_ROOT / "12_final_external_case_sanity_review" / "metrics" / "full_geometry_laterality_lcc_review_all_cases.csv"

METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"
PREVIEWS_DIR = OUTPUT_ROOT / "previews_flagged_cases"

# LCC-collapse definition. This matches the logic we discussed:
# raw prediction had some useful overlap, but LCC-selected component almost loses it.
RAW_DICE_MIN_FOR_COLLAPSE = 0.20
LCC_DICE_MAX_FOR_COLLAPSE = 0.05

# Large spatial displacement threshold.
LARGE_CENTROID_DISTANCE_MM = 100.0

# Midline/side heuristic in voxel index space. If centroid is within this margin of image center,
# it is labeled midline_uncertain, not left or right.
MIDLINE_MARGIN_FRACTION = 0.05
MIN_MIDLINE_MARGIN_VOX = 8

# Bilateral FOV heuristic is intentionally rough. It is only used to decide whether an
# opposite-half flag is anatomically interpretable, not as a paper-level metric.
RUN_BILATERAL_FOV_HEURISTIC_FOR_FLAGGED_CASES = True
MIN_FOREGROUND_VOXELS_PER_HALF = 5000
MIN_LEFT_RIGHT_FOREGROUND_RATIO = 0.10

# Save overlays for a limited number of flagged cases.
MAX_PREVIEW_CASES = 60


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_dirs() -> None:
    for p in [OUTPUT_ROOT, METRICS_DIR, FIGURES_DIR, PREVIEWS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def standardize_case_id(x: str) -> str:
    return str(x).strip().replace(".nii.gz", "").replace(".nii", "").upper()


def cohort_from_case_id(case_id: str) -> str:
    u = standardize_case_id(case_id)
    if u.startswith("DUKE"):
        return "DUKE"
    if u.startswith("ISPY1"):
        return "ISPY1"
    if u.startswith("NACT"):
        return "NACT"
    return "OTHER"


def find_existing_file(directory: Path, case_id: str, prefer_lower: bool = False) -> Optional[Path]:
    case_id = standardize_case_id(case_id)
    candidates = []
    if prefer_lower:
        candidates.extend([case_id.lower(), case_id.upper(), case_id])
    else:
        candidates.extend([case_id.upper(), case_id, case_id.lower()])

    # Try exact names first.
    for base in candidates:
        for suffix in [".nii.gz", ".nii"]:
            p = directory / f"{base}{suffix}"
            if p.exists():
                return p

    # Fallback glob, case-insensitive.
    if directory.exists():
        target = case_id.lower()
        for p in directory.glob("*.nii*"):
            name = p.name.lower().replace(".nii.gz", "").replace(".nii", "")
            if name == target:
                return p
    return None


def load_image_and_binary(path: Path) -> Tuple[sitk.Image, np.ndarray]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, (arr > 0).astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    prec = float(tp / pred.sum()) if pred.sum() > 0 else 0.0
    rec = float(tp / gt.sum()) if gt.sum() > 0 else 1.0 if pred.sum() == 0 else 0.0
    return prec, rec


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(np.uint8)


def n_components(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    _, n = ndimage.label(mask.astype(np.uint8))
    return int(n)


def component_sizes(mask: np.ndarray) -> List[int]:
    if mask.sum() == 0:
        return []
    labeled, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return []
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    return sorted([int(x) for x in sizes], reverse=True)


def centroid_zyx(mask: np.ndarray) -> Optional[np.ndarray]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)  # z,y,x in voxel index coordinates


def centroid_distance_mm(c1: Optional[np.ndarray], c2: Optional[np.ndarray], spacing_xyz: Tuple[float, float, float]) -> float:
    if c1 is None or c2 is None:
        return float("nan")
    # coords are z,y,x; spacing is x,y,z
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    return float(np.linalg.norm((c1 - c2) * spacing_zyx))


def side_label_from_centroid_x(x_index: Optional[float], width: int) -> str:
    if x_index is None or not np.isfinite(x_index):
        return "empty"
    mid = (width - 1) / 2.0
    margin = max(MIN_MIDLINE_MARGIN_VOX, MIDLINE_MARGIN_FRACTION * width)
    if x_index < mid - margin:
        return "left_half"
    if x_index > mid + margin:
        return "right_half"
    return "midline_uncertain"


def opposite_half(side_a: str, side_b: str) -> bool:
    return (side_a in {"left_half", "right_half"} and
            side_b in {"left_half", "right_half"} and
            side_a != side_b)


def geometry_match(gt_img: sitk.Image, pred_img: sitk.Image) -> Dict[str, bool]:
    def close_tuple(a, b, tol=1e-3):
        return bool(np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=tol, rtol=0))
    return {
        "shape_match": gt_img.GetSize() == pred_img.GetSize(),
        "spacing_match": close_tuple(gt_img.GetSpacing(), pred_img.GetSpacing()),
        "origin_match": close_tuple(gt_img.GetOrigin(), pred_img.GetOrigin()),
        "direction_match": close_tuple(gt_img.GetDirection(), pred_img.GetDirection()),
    }


def get_precontrast_image_path(case_id: str) -> Optional[Path]:
    case_dir = IMAGES_DIR / standardize_case_id(case_id)
    if not case_dir.exists():
        # Try lowercase/other variants.
        for d in IMAGES_DIR.glob("*") if IMAGES_DIR.exists() else []:
            if d.is_dir() and standardize_case_id(d.name) == standardize_case_id(case_id):
                case_dir = d
                break
    if not case_dir.exists():
        return None
    nii_files = sorted(list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii")))
    return nii_files[0] if nii_files else None


def bilateral_fov_heuristic(case_id: str) -> Dict[str, object]:
    """Roughly estimate whether both image halves contain meaningful foreground signal."""
    path = get_precontrast_image_path(case_id)
    if path is None:
        return {
            "precontrast_found": False,
            "bilateral_fov_heuristic": "unknown_no_precontrast",
            "foreground_left_voxels": np.nan,
            "foreground_right_voxels": np.nan,
            "left_right_foreground_ratio": np.nan,
        }
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        finite = np.isfinite(arr)
        vals = np.abs(arr[finite])
        vals = vals[vals > 0]
        if len(vals) == 0:
            thr = 0.0
        else:
            thr = max(1e-6, float(np.percentile(vals, 1)))
        fg = np.abs(arr) > thr
        width = fg.shape[-1]
        mid = width // 2
        left = int(fg[..., :mid].sum())
        right = int(fg[..., mid:].sum())
        mn, mx = min(left, right), max(left, right)
        ratio = float(mn / mx) if mx > 0 else 0.0
        bilateral = (left >= MIN_FOREGROUND_VOXELS_PER_HALF and
                     right >= MIN_FOREGROUND_VOXELS_PER_HALF and
                     ratio >= MIN_LEFT_RIGHT_FOREGROUND_RATIO)
        return {
            "precontrast_found": True,
            "bilateral_fov_heuristic": "bilateral_like" if bilateral else "unilateral_or_asymmetric_like",
            "foreground_left_voxels": left,
            "foreground_right_voxels": right,
            "left_right_foreground_ratio": ratio,
        }
    except Exception as e:
        return {
            "precontrast_found": True,
            "bilateral_fov_heuristic": f"error:{type(e).__name__}",
            "foreground_left_voxels": np.nan,
            "foreground_right_voxels": np.nan,
            "left_right_foreground_ratio": np.nan,
        }


# =============================================================================
# Main independent audit
# =============================================================================

def run_independent_audit(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    case_col = "case_id_std" if "case_id_std" in case_df.columns else "case_id"
    if case_col not in case_df.columns:
        raise ValueError("Could not find case_id_std or case_id in case table.")

    for i, row in case_df.iterrows():
        case_id = standardize_case_id(row[case_col])
        cohort = row.get("cohort", cohort_from_case_id(case_id))

        gt_path = find_existing_file(GT_DIR, case_id, prefer_lower=True)
        pred_path = find_existing_file(PRED_V2_DIR, case_id, prefer_lower=False)

        out = {
            "case_id_std": case_id,
            "cohort": cohort,
            "gt_path_found": gt_path is not None,
            "pred_v2_path_found": pred_path is not None,
        }

        if gt_path is None or pred_path is None:
            out.update({"audit_error": "missing_gt_or_prediction"})
            rows.append(out)
            continue

        try:
            gt_img, gt = load_image_and_binary(gt_path)
            pred_img, raw = load_image_and_binary(pred_path)
            geom = geometry_match(gt_img, pred_img)
            out.update(geom)
            out["any_geometry_mismatch"] = not all(geom.values())

            if gt.shape != raw.shape:
                out.update({"audit_error": f"shape_mismatch_array_gt{gt.shape}_pred{raw.shape}"})
                rows.append(out)
                continue

            lcc = keep_largest_component(raw)
            raw_d = dice(raw, gt)
            lcc_d = dice(lcc, gt)
            raw_prec, raw_rec = precision_recall(raw, gt)
            lcc_prec, lcc_rec = precision_recall(lcc, gt)

            gt_c = centroid_zyx(gt)
            raw_c = centroid_zyx(raw)
            lcc_c = centroid_zyx(lcc)
            spacing = gt_img.GetSpacing()
            width = gt.shape[-1]

            gt_side = side_label_from_centroid_x(None if gt_c is None else float(gt_c[2]), width)
            raw_side = side_label_from_centroid_x(None if raw_c is None else float(raw_c[2]), width)
            lcc_side = side_label_from_centroid_x(None if lcc_c is None else float(lcc_c[2]), width)

            raw_sizes = component_sizes(raw)
            lcc_sizes = component_sizes(lcc)
            raw_n = len(raw_sizes)
            raw_sum = int(raw.sum())
            lcc_sum = int(lcc.sum())
            gt_sum = int(gt.sum())

            out.update({
                "audit_error": "",
                "gt_voxels": gt_sum,
                "v2_raw_voxels": raw_sum,
                "v2_lcc_voxels": lcc_sum,
                "v2_raw_n_components_recomputed": raw_n,
                "v2_raw_largest_component_voxels": raw_sizes[0] if raw_sizes else 0,
                "v2_raw_second_largest_component_voxels": raw_sizes[1] if len(raw_sizes) > 1 else 0,
                "v2_raw_largest_component_fraction_recomputed": (raw_sizes[0] / raw_sum) if raw_sum > 0 and raw_sizes else np.nan,
                "v2_raw_second_largest_component_fraction_recomputed": (raw_sizes[1] / raw_sum) if raw_sum > 0 and len(raw_sizes) > 1 else 0.0,
                "v2_raw_dice_recomputed": raw_d,
                "v2_lcc_dice_recomputed": lcc_d,
                "v2_raw_precision_recomputed": raw_prec,
                "v2_raw_recall_recomputed": raw_rec,
                "v2_lcc_precision_recomputed": lcc_prec,
                "v2_lcc_recall_recomputed": lcc_rec,
                "delta_lcc_minus_raw_dice_recomputed": lcc_d - raw_d,
                "raw_to_lcc_dice_drop_recomputed": raw_d - lcc_d,
                "lcc_collapse_independent": bool(raw_d > RAW_DICE_MIN_FOR_COLLAPSE and lcc_d < LCC_DICE_MAX_FOR_COLLAPSE),
                "lcc_dice_zero_or_nearzero": bool(lcc_d < LCC_DICE_MAX_FOR_COLLAPSE),
                "raw_useful_overlap_before_lcc": bool(raw_d > RAW_DICE_MIN_FOR_COLLAPSE),
                "gt_centroid_z": np.nan if gt_c is None else float(gt_c[0]),
                "gt_centroid_y": np.nan if gt_c is None else float(gt_c[1]),
                "gt_centroid_x": np.nan if gt_c is None else float(gt_c[2]),
                "raw_centroid_z": np.nan if raw_c is None else float(raw_c[0]),
                "raw_centroid_y": np.nan if raw_c is None else float(raw_c[1]),
                "raw_centroid_x": np.nan if raw_c is None else float(raw_c[2]),
                "lcc_centroid_z": np.nan if lcc_c is None else float(lcc_c[0]),
                "lcc_centroid_y": np.nan if lcc_c is None else float(lcc_c[1]),
                "lcc_centroid_x": np.nan if lcc_c is None else float(lcc_c[2]),
                "gt_side_half": gt_side,
                "raw_side_half": raw_side,
                "lcc_side_half": lcc_side,
                "raw_opposite_half_to_gt_independent": bool(opposite_half(gt_side, raw_side)),
                "lcc_opposite_half_to_gt_independent": bool(opposite_half(gt_side, lcc_side)),
                "raw_centroid_distance_to_gt_mm": centroid_distance_mm(raw_c, gt_c, spacing),
                "lcc_centroid_distance_to_gt_mm": centroid_distance_mm(lcc_c, gt_c, spacing),
            })
            out["large_lcc_centroid_distance_gt_100mm_independent"] = bool(
                np.isfinite(out["lcc_centroid_distance_to_gt_mm"]) and
                out["lcc_centroid_distance_to_gt_mm"] > LARGE_CENTROID_DISTANCE_MM
            )
            out["large_raw_centroid_distance_gt_100mm_independent"] = bool(
                np.isfinite(out["raw_centroid_distance_to_gt_mm"]) and
                out["raw_centroid_distance_to_gt_mm"] > LARGE_CENTROID_DISTANCE_MM
            )

            # Compare with table dice if available.
            for src_col, new_col in [
                ("v2_raw_dice", "v2_raw_dice_recomputed"),
                ("v2_lcc_dice", "v2_lcc_dice_recomputed"),
            ]:
                if src_col in row.index:
                    try:
                        out[f"{src_col}_table"] = float(row[src_col])
                        out[f"{src_col}_abs_diff_recomputed"] = abs(float(row[src_col]) - float(out[new_col]))
                    except Exception:
                        out[f"{src_col}_table"] = np.nan
                        out[f"{src_col}_abs_diff_recomputed"] = np.nan

        except Exception as e:
            out.update({"audit_error": repr(e)})

        rows.append(out)
        if (i + 1) % 50 == 0:
            print(f"  independent audit {i+1}/{len(case_df)}")

    df = pd.DataFrame(rows)

    # Optional bilateral FOV only for flagged cases, because it requires loading the raw precontrast image.
    if RUN_BILATERAL_FOV_HEURISTIC_FOR_FLAGGED_CASES and len(df) > 0:
        flagged_mask = (
            df.get("lcc_collapse_independent", False).fillna(False).astype(bool) |
            df.get("lcc_opposite_half_to_gt_independent", False).fillna(False).astype(bool) |
            df.get("large_lcc_centroid_distance_gt_100mm_independent", False).fillna(False).astype(bool)
        )
        fov_rows = []
        flagged_ids = df.loc[flagged_mask, "case_id_std"].tolist()
        print(f"  bilateral FOV heuristic for flagged cases: {len(flagged_ids)}")
        for case_id in flagged_ids:
            d = {"case_id_std": case_id}
            d.update(bilateral_fov_heuristic(case_id))
            fov_rows.append(d)
        if fov_rows:
            fov_df = pd.DataFrame(fov_rows)
            df = df.merge(fov_df, on="case_id_std", how="left")
        else:
            df["bilateral_fov_heuristic"] = np.nan
    return df


# =============================================================================
# Existing-flag comparison and summaries
# =============================================================================

def make_flag_set(df: pd.DataFrame, col: str) -> set:
    if col not in df.columns:
        return set()
    return set(df.loc[df[col].fillna(False).astype(bool), "case_id_std"].astype(str))


def save_flag_crosstabs(audit: pd.DataFrame) -> Dict[str, object]:
    collapse_col = "lcc_collapse_independent"
    opp_col = "lcc_opposite_half_to_gt_independent"
    large_col = "large_lcc_centroid_distance_gt_100mm_independent"

    for c in [collapse_col, opp_col, large_col]:
        if c not in audit.columns:
            audit[c] = False
        audit[c] = audit[c].fillna(False).astype(bool)

    # Two-way crosstabs.
    ct_collapse_opp = pd.crosstab(audit[collapse_col], audit[opp_col], margins=True)
    ct_collapse_large = pd.crosstab(audit[collapse_col], audit[large_col], margins=True)
    ct_opp_large = pd.crosstab(audit[opp_col], audit[large_col], margins=True)

    ct_collapse_opp.to_csv(METRICS_DIR / "crosstab_lcc_collapse_vs_opposite_half.csv")
    ct_collapse_large.to_csv(METRICS_DIR / "crosstab_lcc_collapse_vs_centroid_gt100mm.csv")
    ct_opp_large.to_csv(METRICS_DIR / "crosstab_opposite_half_vs_centroid_gt100mm.csv")

    # Three-way counts.
    three = (audit.groupby([collapse_col, opp_col, large_col], dropna=False)
             .size().reset_index(name="n"))
    three.to_csv(METRICS_DIR / "crosstab_3way_lcc_collapse_opposite_centroid100.csv", index=False)

    # Set identity/intersections.
    collapse_ids = make_flag_set(audit, collapse_col)
    opp_ids = make_flag_set(audit, opp_col)
    large_ids = make_flag_set(audit, large_col)
    stats_out = {
        "n_all": len(audit),
        "n_lcc_collapse": len(collapse_ids),
        "n_lcc_opposite_half": len(opp_ids),
        "n_large_lcc_centroid_gt100mm": len(large_ids),
        "intersection_collapse_opposite": len(collapse_ids & opp_ids),
        "intersection_collapse_large100": len(collapse_ids & large_ids),
        "intersection_opposite_large100": len(opp_ids & large_ids),
        "intersection_all_three": len(collapse_ids & opp_ids & large_ids),
        "collapse_equals_opposite_set": collapse_ids == opp_ids,
        "collapse_subset_of_opposite": collapse_ids.issubset(opp_ids),
        "opposite_subset_of_collapse": opp_ids.issubset(collapse_ids),
        "opposite_outside_collapse": len(opp_ids - collapse_ids),
        "collapse_outside_opposite": len(collapse_ids - opp_ids),
        "large100_outside_collapse": len(large_ids - collapse_ids),
    }
    pd.DataFrame([stats_out]).to_csv(METRICS_DIR / "flag_set_intersection_summary.csv", index=False)

    # Cohort distribution.
    cohort = []
    for coh, grp in audit.groupby("cohort"):
        n = len(grp)
        cohort.append({
            "cohort": coh,
            "n": n,
            "lcc_collapse_n": int(grp[collapse_col].sum()),
            "lcc_collapse_rate": float(grp[collapse_col].mean()) if n else np.nan,
            "lcc_opposite_half_n": int(grp[opp_col].sum()),
            "lcc_opposite_half_rate": float(grp[opp_col].mean()) if n else np.nan,
            "large_lcc_centroid_gt100_n": int(grp[large_col].sum()),
            "large_lcc_centroid_gt100_rate": float(grp[large_col].mean()) if n else np.nan,
            "mean_raw_dice": float(grp["v2_raw_dice_recomputed"].mean()),
            "mean_lcc_dice": float(grp["v2_lcc_dice_recomputed"].mean()),
            "mean_delta_lcc_minus_raw": float(grp["delta_lcc_minus_raw_dice_recomputed"].mean()),
        })
    cohort_df = pd.DataFrame(cohort)
    cohort_df.to_csv(METRICS_DIR / "lcc_side_flags_by_cohort.csv", index=False)

    # Case ID lists.
    def save_case_list(ids: set, name: str):
        sub = audit[audit["case_id_std"].isin(ids)].copy()
        cols = [c for c in [
            "case_id_std", "cohort", "v2_raw_dice_recomputed", "v2_lcc_dice_recomputed",
            "delta_lcc_minus_raw_dice_recomputed", "gt_side_half", "raw_side_half", "lcc_side_half",
            "raw_centroid_distance_to_gt_mm", "lcc_centroid_distance_to_gt_mm",
            "bilateral_fov_heuristic", "foreground_left_voxels", "foreground_right_voxels",
            "left_right_foreground_ratio",
        ] if c in sub.columns]
        sub[cols].sort_values(["cohort", "case_id_std"]).to_csv(METRICS_DIR / name, index=False)

    save_case_list(collapse_ids, "lcc_collapse_case_list.csv")
    save_case_list(opp_ids, "opposite_half_case_list.csv")
    save_case_list(large_ids, "large_lcc_centroid_gt100mm_case_list.csv")
    save_case_list(collapse_ids & opp_ids & large_ids, "intersection_all_three_case_list.csv")

    return stats_out


def compare_with_existing_12(audit: pd.DataFrame) -> None:
    if not EXISTING_12_REVIEW.exists():
        return
    try:
        old = pd.read_csv(EXISTING_12_REVIEW)
        case_col = "case_id_std" if "case_id_std" in old.columns else "case_id"
        if case_col not in old.columns:
            return
        old["case_id_std"] = old[case_col].apply(standardize_case_id)
        merged = audit.merge(old, on="case_id_std", how="left", suffixes=("_independent", "_existing12"))
        merged.to_csv(METRICS_DIR / "comparison_with_existing_12_review.csv", index=False)

        # Find likely old flag columns and summarize overlap.
        rows = []
        old_cols = [c for c in old.columns if any(s in c.lower() for s in ["opposite", "collapse", "centroid", "geometry"])]
        for c in old_cols:
            vals = old[c]
            if vals.dtype == bool or set(vals.dropna().unique()).issubset({True, False, 0, 1, "True", "False", "true", "false"}):
                try:
                    old_set = set(old.loc[old[c].astype(str).str.lower().isin(["true", "1"]), "case_id_std"])
                    for new_c in ["lcc_collapse_independent", "lcc_opposite_half_to_gt_independent", "large_lcc_centroid_gt_100mm_independent"]:
                        new_set = make_flag_set(audit, new_c)
                        rows.append({
                            "existing12_col": c,
                            "independent_col": new_c,
                            "n_existing12_true": len(old_set),
                            "n_independent_true": len(new_set),
                            "intersection": len(old_set & new_set),
                            "existing_subset_of_independent": old_set.issubset(new_set),
                            "independent_subset_of_existing": new_set.issubset(old_set),
                            "sets_equal": old_set == new_set,
                        })
                except Exception:
                    pass
        if rows:
            pd.DataFrame(rows).to_csv(METRICS_DIR / "existing12_flag_overlap_summary.csv", index=False)
    except Exception as e:
        with open(METRICS_DIR / "existing12_comparison_error.txt", "w", encoding="utf-8") as f:
            f.write(repr(e))


# =============================================================================
# Cleaner cross-family collinearity table
# =============================================================================

def family_and_base(col: str) -> Tuple[str, str]:
    c = str(col)
    lc = c.lower()

    # Normalize prefixes and self-transform variants.
    base = lc
    for p in ["v1_raw_", "v2_raw_", "v1_lcc_", "v2_lcc_", "v1_small50_", "v2_small50_", "v1_small100_", "v2_small100_", "kin_"]:
        base = base.replace(p, "")
    for p in ["gt_", "fg_", "ring_", "tumor_to_ring_", "tumor_to_fg_"]:
        if base.startswith(p):
            base = base[len(p):]
    base = base.replace("log_", "")
    # Collapse timing variants that are deterministic / near-deterministic transforms.
    if base in ["first_post_s", "gt_first_post_s", "fg_first_post_s", "ring_first_post_s"]:
        base = "first_post_s"

    if any(k in lc for k in ["dice", "precision", "recall", "iou", "error_1_minus"]):
        fam = "performance"
    elif any(k in lc for k in ["n_components", "largest_component", "second_largest_component", "component_fraction"]):
        fam = "component_structure"
    elif any(k in lc for k in ["pred_vol", "gt_vol", "pred_to_gt", "volume", "vol_ratio", "theoretical_max", "utilization", "gap_to_volume"]):
        fam = "volume"
    elif any(k in lc for k in ["first_post", "early_dev", "late_dev", "last_post"]):
        fam = "timing"
    elif any(k in lc for k in ["n_phases", "phase_quality", "cohort"]):
        fam = "cohort_protocol"
    elif any(k in lc for k in ["enh", "auc", "washout", "slope"]):
        fam = "contrast_kinetic"
    else:
        fam = "other"
    return fam, base


def clean_cross_family_correlations(case_df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    # Merge useful independent audit columns into case table.
    cols_to_merge = [
        "case_id_std", "v2_raw_n_components_recomputed", "v2_raw_largest_component_fraction_recomputed",
        "v2_raw_second_largest_component_fraction_recomputed", "v2_raw_voxels", "v2_lcc_voxels",
        "v2_raw_dice_recomputed", "v2_lcc_dice_recomputed", "lcc_centroid_distance_to_gt_mm",
    ]
    m = case_df.copy()
    if "case_id_std" not in m.columns:
        case_col = "case_id" if "case_id" in m.columns else m.columns[0]
        m["case_id_std"] = m[case_col].apply(standardize_case_id)
    add = audit[[c for c in cols_to_merge if c in audit.columns]].copy()
    m = m.merge(add, on="case_id_std", how="left")

    numeric_cols = []
    for c in m.columns:
        if c == "case_id_std":
            continue
        try:
            ser = pd.to_numeric(m[c], errors="coerce")
            if ser.notna().sum() >= 30 and ser.nunique(dropna=True) >= 3:
                numeric_cols.append(c)
                m[c] = ser
        except Exception:
            pass

    # Exclude performance columns from predictor-predictor collinearity.
    features = []
    for c in numeric_cols:
        fam, base = family_and_base(c)
        if fam != "performance":
            features.append(c)

    rows = []
    for i, a in enumerate(features):
        fam_a, base_a = family_and_base(a)
        for b in features[i + 1:]:
            fam_b, base_b = family_and_base(b)
            if fam_a == fam_b:
                continue
            # Drop near-duplicate/self-transform versions even across family labels.
            if base_a == base_b:
                continue
            sub = m[[a, b]].dropna()
            if len(sub) < 30 or sub[a].nunique() < 3 or sub[b].nunique() < 3:
                continue
            rho, p = stats.spearmanr(sub[a], sub[b])
            if not np.isfinite(rho):
                continue
            rows.append({
                "feature_a": a,
                "feature_b": b,
                "family_a": fam_a,
                "family_b": fam_b,
                "base_a": base_a,
                "base_b": base_b,
                "spearman_rho": float(rho),
                "abs_rho": float(abs(rho)),
                "p_value": float(p),
                "n": len(sub),
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = out.sort_values("abs_rho", ascending=False)
    # BH correction.
    pvals = out["p_value"].values
    order = np.argsort(pvals)
    ranked = np.empty_like(order, dtype=float)
    m_tests = len(pvals)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        # reverse BH cumulative min
        true_rank = m_tests - rank + 1
        val = pvals[idx] * m_tests / true_rank
        prev = min(prev, val)
        ranked[idx] = min(prev, 1.0)
    out["q_value_bh"] = ranked
    out.to_csv(METRICS_DIR / "cleaned_cross_family_predictor_correlations_strict.csv", index=False)
    return out


# =============================================================================
# Figures and previews
# =============================================================================

def make_figures(audit: pd.DataFrame) -> None:
    if plt is None:
        return
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180})

    # LCC collapse rate by cohort.
    cohort = pd.read_csv(METRICS_DIR / "lcc_side_flags_by_cohort.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cohort["cohort"], cohort["lcc_collapse_rate"])
    ax.set_ylabel("LCC-collapse rate")
    ax.set_title("LCC-collapse rate by external cohort")
    for i, r in cohort.iterrows():
        ax.text(i, r["lcc_collapse_rate"] + 0.005, f"{int(r['lcc_collapse_n'])}/{int(r['n'])}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_lcc_collapse_rate_by_cohort.png")
    plt.close(fig)

    # Raw vs LCC Dice scatter.
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(audit["lcc_collapse_independent"].fillna(False).astype(bool), "tab:red", "tab:blue")
    ax.scatter(audit["v2_raw_dice_recomputed"], audit["v2_lcc_dice_recomputed"], s=14, alpha=0.7, c=colors)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("v2 raw Dice")
    ax.set_ylabel("v2 + LCC Dice")
    ax.set_title("Case-level LCC trade-off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_raw_vs_lcc_dice_collapse_highlight.png")
    plt.close(fig)

    # Centroid distance histogram.
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = audit["lcc_centroid_distance_to_gt_mm"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(vals, bins=40)
    ax.axvline(LARGE_CENTROID_DISTANCE_MM, linestyle="--", linewidth=1)
    ax.set_xlabel("LCC centroid distance to GT (mm)")
    ax.set_ylabel("Cases")
    ax.set_title("Spatial displacement after LCC")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_lcc_centroid_distance_histogram.png")
    plt.close(fig)

    # 2x2 crosstab heatmap collapse vs opposite.
    ct = pd.crosstab(audit["lcc_collapse_independent"], audit["lcc_opposite_half_to_gt_independent"])
    fig, ax = plt.subplots(figsize=(5, 4))
    mat = ct.reindex(index=[False, True], columns=[False, True], fill_value=0).values
    im = ax.imshow(mat)
    ax.set_xticks([0, 1], labels=["Opposite no", "Opposite yes"])
    ax.set_yticks([0, 1], labels=["Collapse no", "Collapse yes"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_title("LCC-collapse vs opposite-half flag")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_crosstab_collapse_vs_opposite_half.png")
    plt.close(fig)


def save_preview_for_case(case_id: str) -> Optional[Path]:
    if plt is None:
        return None
    gt_path = find_existing_file(GT_DIR, case_id, prefer_lower=True)
    pred_path = find_existing_file(PRED_V2_DIR, case_id, prefer_lower=False)
    pre_path = get_precontrast_image_path(case_id)
    if gt_path is None or pred_path is None:
        return None
    try:
        _, gt = load_image_and_binary(gt_path)
        _, raw = load_image_and_binary(pred_path)
        lcc = keep_largest_component(raw)
        if pre_path and pre_path.exists():
            pre_img = sitk.ReadImage(str(pre_path))
            img_arr = sitk.GetArrayFromImage(pre_img).astype(np.float32)
        else:
            img_arr = raw.astype(np.float32)

        # choose slice where GT exists, otherwise raw/lcc maximum.
        sums = gt.sum(axis=(1, 2))
        if sums.max() > 0:
            z = int(np.argmax(sums))
        else:
            sums = raw.sum(axis=(1, 2)) + lcc.sum(axis=(1, 2))
            z = int(np.argmax(sums)) if sums.max() > 0 else img_arr.shape[0] // 2

        base = img_arr[z]
        p1, p99 = np.percentile(base[np.isfinite(base)], [1, 99]) if np.isfinite(base).any() else (0, 1)
        if p99 <= p1:
            p99 = p1 + 1
        norm = np.clip((base - p1) / (p99 - p1), 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        panels = [
            (norm, "image"),
            (gt[z], "GT"),
            (raw[z], "raw pred"),
            (lcc[z], "LCC pred"),
        ]
        for ax, (arr, title) in zip(axes, panels):
            ax.imshow(norm, cmap="gray")
            if title == "GT":
                ax.contour(gt[z], levels=[0.5], colors=["lime"], linewidths=1.2)
            elif title == "raw pred":
                ax.contour(raw[z], levels=[0.5], colors=["red"], linewidths=1.0)
                ax.contour(gt[z], levels=[0.5], colors=["lime"], linewidths=0.8)
            elif title == "LCC pred":
                ax.contour(lcc[z], levels=[0.5], colors=["yellow"], linewidths=1.0)
                ax.contour(gt[z], levels=[0.5], colors=["lime"], linewidths=0.8)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(case_id)
        fig.tight_layout()
        out = PREVIEWS_DIR / f"{case_id}_audit_preview.png"
        fig.savefig(out)
        plt.close(fig)
        return out
    except Exception:
        return None


def make_selected_previews(audit: pd.DataFrame) -> None:
    if plt is None:
        return
    # Priority: all LCC-collapse, then opposite/large distance cases.
    selected = audit[
        audit["lcc_collapse_independent"].fillna(False).astype(bool) |
        audit["lcc_opposite_half_to_gt_independent"].fillna(False).astype(bool) |
        audit["large_lcc_centroid_distance_gt_100mm_independent"].fillna(False).astype(bool)
    ].copy()
    selected = selected.sort_values(["lcc_collapse_independent", "lcc_centroid_distance_to_gt_mm"], ascending=[False, False])
    selected = selected.head(MAX_PREVIEW_CASES)
    rows = []
    for case_id in selected["case_id_std"].tolist():
        p = save_preview_for_case(case_id)
        rows.append({"case_id_std": case_id, "preview_path": str(p) if p else ""})
    if rows:
        pd.DataFrame(rows).to_csv(METRICS_DIR / "preview_file_list.csv", index=False)


# =============================================================================
# Takeaways
# =============================================================================

def write_takeaways(audit: pd.DataFrame, stats_out: Dict[str, object], clean_corr: pd.DataFrame) -> None:
    cohort = pd.read_csv(METRICS_DIR / "lcc_side_flags_by_cohort.csv")
    lines = []
    lines.append("13 FINAL LCC / SIDE-FLAG AUDIT — QUICK TAKEAWAYS")
    lines.append("=" * 78)
    lines.append(f"Cases audited independently: {len(audit)}")
    lines.append("")
    lines.append("Independent flags over all external cases:")
    lines.append(f"  LCC-collapse cases: {stats_out['n_lcc_collapse']} / {stats_out['n_all']} ({stats_out['n_lcc_collapse']/stats_out['n_all']:.1%})")
    lines.append(f"  LCC opposite-half flags: {stats_out['n_lcc_opposite_half']} / {stats_out['n_all']} ({stats_out['n_lcc_opposite_half']/stats_out['n_all']:.1%})")
    lines.append(f"  Large LCC centroid distance > {LARGE_CENTROID_DISTANCE_MM:.0f} mm: {stats_out['n_large_lcc_centroid_gt100mm']} / {stats_out['n_all']} ({stats_out['n_large_lcc_centroid_gt100mm']/stats_out['n_all']:.1%})")
    lines.append("")
    lines.append("Set-overlap checks:")
    lines.append(f"  collapse ∩ opposite: {stats_out['intersection_collapse_opposite']}")
    lines.append(f"  collapse ∩ centroid>100mm: {stats_out['intersection_collapse_large100']}")
    lines.append(f"  opposite ∩ centroid>100mm: {stats_out['intersection_opposite_large100']}")
    lines.append(f"  all three: {stats_out['intersection_all_three']}")
    lines.append(f"  collapse set equals opposite set: {stats_out['collapse_equals_opposite_set']}")
    lines.append(f"  opposite flags outside collapse: {stats_out['opposite_outside_collapse']}")
    lines.append(f"  collapse cases outside opposite: {stats_out['collapse_outside_opposite']}")
    lines.append("")
    lines.append("Cohort distribution:")
    for _, r in cohort.iterrows():
        lines.append(
            f"  {r['cohort']}: collapse {int(r['lcc_collapse_n'])}/{int(r['n'])} "
            f"({r['lcc_collapse_rate']:.1%}), opposite {int(r['lcc_opposite_half_n'])}/{int(r['n'])} "
            f"({r['lcc_opposite_half_rate']:.1%}), centroid>100mm {int(r['large_lcc_centroid_gt100_n'])}/{int(r['n'])} "
            f"({r['large_lcc_centroid_gt100_rate']:.1%})"
        )
    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("  - If collapse and opposite-half sets are identical, check whether this is a true anatomical pattern or a definition artifact.")
    lines.append("  - Opposite-half is an image-half heuristic, not a definitive anatomical laterality diagnosis.")
    lines.append("  - Use 'large spatial displacement / wrong-component selection' in the manuscript unless manual review confirms true contralateral selection.")
    lines.append("  - Geometry mismatch should be reported separately from centroid/side flags.")
    lines.append("")
    if "any_geometry_mismatch" in audit.columns:
        n_geom = int(audit["any_geometry_mismatch"].fillna(False).astype(bool).sum())
        lines.append(f"Geometry mismatch cases: {n_geom}")
        if n_geom == 0:
            lines.append("  No shape/spacing/origin/direction mismatch was detected in the audited external predictions.")
    lines.append("")
    if len(clean_corr) > 0:
        lines.append("Top strict cleaned cross-family correlations:")
        for _, r in clean_corr.head(10).iterrows():
            lines.append(f"  {r['feature_a']} vs {r['feature_b']}: rho={r['spearman_rho']:.3f} ({r['family_a']} vs {r['family_b']})")
    lines.append("")
    lines.append("Paper-safe wording:")
    lines.append("  Uniform LCC postprocessing improved mean external Dice but produced an identifiable trade-off: a subset of cases showed LCC-collapse, where the raw prediction had partial tumor overlap but the selected largest component had near-zero overlap. Image-geometry checks did not reveal a global affine/spacing/orientation mismatch. These findings support reporting both raw and postprocessed external results and treating morphology-only postprocessing as a sensitivity analysis rather than a clinically safe correction.")

    with open(OUTPUT_ROOT / "article_takeaways_13.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Entrypoint
# =============================================================================

def main() -> None:
    print("13 Final LCC/side-flag audit")
    print("=" * 36)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Case table  : {CASE_TABLE}")
    ensure_dirs()

    if not CASE_TABLE.exists():
        raise FileNotFoundError(f"Case table not found: {CASE_TABLE}")
    case_df = pd.read_csv(CASE_TABLE)
    if "case_id_std" not in case_df.columns:
        case_col = "case_id" if "case_id" in case_df.columns else case_df.columns[0]
        case_df["case_id_std"] = case_df[case_col].apply(standardize_case_id)
    if "cohort" not in case_df.columns:
        case_df["cohort"] = case_df["case_id_std"].apply(cohort_from_case_id)

    print(f"Cases loaded: {len(case_df)}")
    print("Running independent all-case audit...")
    audit = run_independent_audit(case_df)
    audit_path = METRICS_DIR / "independent_lcc_side_audit_all_cases.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Saved independent audit: {audit_path}")

    print("Writing crosstabs and cohort summaries...")
    stats_out = save_flag_crosstabs(audit)

    print("Comparing with existing 12 flags if available...")
    compare_with_existing_12(audit)

    print("Computing stricter cleaned cross-family correlations...")
    clean_corr = clean_cross_family_correlations(case_df, audit)

    print("Creating figures and previews...")
    make_figures(audit)
    make_selected_previews(audit)

    write_takeaways(audit, stats_out, clean_corr)

    print("\nDONE — 13 final audit complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Previews: {PREVIEWS_DIR}")
    print(f"Takeaways: {OUTPUT_ROOT / 'article_takeaways_13.txt'}")
    print("\nQuick results:")
    print(f"  all cases audited: {stats_out['n_all']}")
    print(f"  LCC-collapse: {stats_out['n_lcc_collapse']}")
    print(f"  opposite-half flags: {stats_out['n_lcc_opposite_half']}")
    print(f"  centroid >100mm: {stats_out['n_large_lcc_centroid_gt100mm']}")
    print(f"  collapse ∩ opposite: {stats_out['intersection_collapse_opposite']}")
    print(f"  collapse set == opposite set: {stats_out['collapse_equals_opposite_set']}")
    print(f"  opposite outside collapse: {stats_out['opposite_outside_collapse']}")
    print(f"  collapse outside opposite: {stats_out['collapse_outside_opposite']}")
    if len(clean_corr) > 0:
        print("\nTop strict cleaned cross-family correlations:")
        print(clean_corr[["feature_a", "feature_b", "family_a", "family_b", "spearman_rho", "abs_rho"]].head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        raise
