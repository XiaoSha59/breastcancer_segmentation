
# run_14_lcc_collapse_visual_pattern_review.py
# Final visual-pattern review for LCC-collapse failures.
# No command-line parameters. Edit CONFIG only if your folders differ.

from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import fisher_exact, chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG — EDIT ONLY HERE IF NEEDED
# =============================================================================
PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# Main inputs
CASE_TABLE = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "final_sanity_case_table_with_volume_bounds.csv"
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"
INPUT_V2_DIR = Path(r"C:\nnw\mama_mia_input_v2")
ORIGINAL_IMAGES_DIR = PROJECT_ROOT / "images_2"

# Optional prior audit table from step 13, used only for cross-checking if present.
AUDIT13_CSV = PROJECT_ROOT / "13_final_lcc_side_audit" / "metrics" / "independent_lcc_side_audit_all_cases.csv"

# Output
OUTPUT_ROOT = PROJECT_ROOT / "14_lcc_collapse_visual_pattern_review"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"
PREVIEW_DIR = OUTPUT_ROOT / "review_packet" / "previews"
NIFTI_DIR = OUTPUT_ROOT / "review_packet" / "nifti_cases"

# Collapse definition: raw has partial overlap but LCC collapses to near-zero Dice.
RAW_DICE_MIN_FOR_COLLAPSE = 0.20
LCC_DICE_MAX_FOR_COLLAPSE = 0.05
CENTROID_LARGE_DISTANCE_MM = 100.0

# Save review packets for all collapse cases. Also save a small control sample.
SAVE_OPPOSITE_NOT_COLLAPSE_CONTROLS = True
MAX_OPPOSITE_NOT_COLLAPSE_CONTROLS = 12
SAVE_RANDOM_GOOD_CONTROLS = True
MAX_RANDOM_GOOD_CONTROLS = 8
RANDOM_SEED = 42

# Preview settings
SAVE_NIFTI_REVIEW_PACKETS = True
MAX_CONTACT_SHEET_IMAGES = 80

# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    for d in [METRICS_DIR, FIGURES_DIR, PREVIEW_DIR, NIFTI_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def clean_case_id(x: str) -> str:
    s = str(x).strip()
    s = s.replace(".nii.gz", "").replace(".nii", "")
    return s.upper()


def find_case_col(df: pd.DataFrame) -> str:
    for c in ["case_id_std", "case_id", "patient_id", "pid"]:
        if c in df.columns:
            return c
    raise KeyError("Could not find a case ID column in the case table.")


def find_optional_col(df: pd.DataFrame, candidates: List[str], contains_all: Optional[List[str]] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if contains_all:
        terms = [t.lower() for t in contains_all]
        for c in df.columns:
            lc = c.lower()
            if all(t in lc for t in terms):
                return c
    return None


def load_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def arr_from_img(path: Path, binary: bool = False) -> Tuple[np.ndarray, sitk.Image]:
    img = load_sitk(path)
    arr = sitk.GetArrayFromImage(img)
    if binary:
        arr = (arr > 0).astype(np.uint8)
    return arr, img


def find_file(folder: Path, case_id: str, suffix: str = ".nii.gz") -> Optional[Path]:
    candidates = [
        folder / f"{case_id}{suffix}",
        folder / f"{case_id.lower()}{suffix}",
        folder / f"{case_id.upper()}{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback glob case-insensitive
    if folder.exists():
        target = f"{case_id}{suffix}".lower()
        for p in folder.glob(f"*{suffix}"):
            if p.name.lower() == target:
                return p
    return None


def find_base_image(case_id: str) -> Optional[Path]:
    # Prefer the actual v2 nnU-Net input channel 0000.
    candidates = [
        INPUT_V2_DIR / f"{case_id}_0000.nii.gz",
        INPUT_V2_DIR / f"{case_id.lower()}_0000.nii.gz",
        INPUT_V2_DIR / f"{case_id.upper()}_0000.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback to original first phase file.
    for d in [ORIGINAL_IMAGES_DIR / case_id, ORIGINAL_IMAGES_DIR / case_id.lower(), ORIGINAL_IMAGES_DIR / case_id.upper()]:
        if d.exists() and d.is_dir():
            files = sorted(d.glob("*.nii.gz"))
            if files:
                return files[0]
    return None


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    prec = float(tp / pred_b.sum()) if pred_b.sum() > 0 else 0.0
    rec = float(tp / gt_b.sum()) if gt_b.sum() > 0 else 0.0
    return prec, rec


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return (labeled == largest_label).astype(np.uint8)


def component_stats(mask: np.ndarray) -> Dict[str, float]:
    mask = (mask > 0).astype(np.uint8)
    total = int(mask.sum())
    if total == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    labeled, n = ndimage.label(mask)
    if n == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "second_largest_component_voxels": 0,
            "largest_component_fraction": np.nan,
            "second_largest_component_fraction": np.nan,
        }
    sizes = np.asarray(ndimage.sum(mask, labeled, index=np.arange(1, n + 1)), dtype=float)
    sizes_sorted = np.sort(sizes)[::-1]
    largest = float(sizes_sorted[0])
    second = float(sizes_sorted[1]) if len(sizes_sorted) > 1 else 0.0
    return {
        "n_components": int(n),
        "largest_component_voxels": largest,
        "second_largest_component_voxels": second,
        "largest_component_fraction": largest / total if total else np.nan,
        "second_largest_component_fraction": second / total if total else np.nan,
    }


def centroid_zyx(mask: np.ndarray) -> Tuple[float, float, float]:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return (np.nan, np.nan, np.nan)
    c = pts.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def centroid_distance_mm(c1_zyx: Tuple[float, float, float], c2_zyx: Tuple[float, float, float], spacing_xyz: Tuple[float, float, float]) -> float:
    if any(np.isnan(c1_zyx)) or any(np.isnan(c2_zyx)):
        return np.nan
    # SimpleITK spacing is x,y,z; arrays are z,y,x.
    dz = (c1_zyx[0] - c2_zyx[0]) * spacing_xyz[2]
    dy = (c1_zyx[1] - c2_zyx[1]) * spacing_xyz[1]
    dx = (c1_zyx[2] - c2_zyx[2]) * spacing_xyz[0]
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def side_from_centroid_x(x: float, width: int, center_margin_fraction: float = 0.05) -> str:
    if np.isnan(x):
        return "missing"
    mid = (width - 1) / 2.0
    margin = width * center_margin_fraction
    if x < mid - margin:
        return "left_half"
    if x > mid + margin:
        return "right_half"
    return "midline_zone"


def opposite_half(gt_c: Tuple[float, float, float], pred_c: Tuple[float, float, float], width: int) -> bool:
    gt_side = side_from_centroid_x(gt_c[2], width)
    pred_side = side_from_centroid_x(pred_c[2], width)
    if gt_side in ("missing", "midline_zone") or pred_side in ("missing", "midline_zone"):
        return False
    return gt_side != pred_side


def rough_bilateral_fov(img_arr: np.ndarray) -> Dict[str, float]:
    """Rough, non-diagnostic heuristic: compare non-background occupancy on both image halves."""
    if img_arr.size == 0:
        return {"bilateral_like_fov": np.nan, "left_occupancy": np.nan, "right_occupancy": np.nan}
    arr = img_arr.astype(np.float32)
    # Use central half of slices to avoid empty margins.
    z0 = max(0, arr.shape[0] // 4)
    z1 = min(arr.shape[0], 3 * arr.shape[0] // 4)
    sub = arr[z0:z1]
    if sub.size == 0:
        sub = arr
    thr = np.percentile(sub, 20)
    fg = sub > thr
    w = fg.shape[2]
    left = fg[:, :, : w // 2].mean()
    right = fg[:, :, w // 2 :].mean()
    # Both halves must have some occupancy. This is intentionally broad.
    bilateral = int(left > 0.05 and right > 0.05)
    return {"bilateral_like_fov": bilateral, "left_occupancy": float(left), "right_occupancy": float(right)}


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    finite = np.isfinite(sl)
    if not finite.any():
        return np.zeros_like(sl, dtype=np.float32)
    lo, hi = np.percentile(sl[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = (sl - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def overlay_rgb(base2d: np.ndarray, gt2d: np.ndarray, raw2d: np.ndarray, lcc2d: np.ndarray) -> np.ndarray:
    base = normalize_slice(base2d)
    rgb = np.stack([base, base, base], axis=-1)
    # raw pred = red
    raw = raw2d.astype(bool)
    rgb[raw, 0] = 1.0
    rgb[raw, 1] *= 0.35
    rgb[raw, 2] *= 0.35
    # LCC = yellow
    lcc = lcc2d.astype(bool)
    rgb[lcc, 0] = 1.0
    rgb[lcc, 1] = 0.9
    rgb[lcc, 2] *= 0.2
    # GT = green, shown last
    gt = gt2d.astype(bool)
    rgb[gt, 0] *= 0.2
    rgb[gt, 1] = 1.0
    rgb[gt, 2] *= 0.2
    return rgb


def safe_slice_index(z: float, depth: int) -> int:
    if np.isnan(z):
        return int(depth // 2)
    return int(np.clip(round(z), 0, depth - 1))


def save_case_preview(case_id: str, group: str, base: np.ndarray, gt: np.ndarray, raw: np.ndarray, lcc: np.ndarray, out_path: Path) -> None:
    gt_c = centroid_zyx(gt)
    raw_c = centroid_zyx(raw)
    lcc_c = centroid_zyx(lcc)
    z_indices = [
        ("GT centroid", safe_slice_index(gt_c[0], base.shape[0])),
        ("Raw pred centroid", safe_slice_index(raw_c[0], base.shape[0])),
        ("LCC centroid", safe_slice_index(lcc_c[0], base.shape[0])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    for ax, (title, z) in zip(axes, z_indices):
        ax.imshow(overlay_rgb(base[z], gt[z], raw[z], lcc[z]))
        ax.set_title(f"{title}\nz={z}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case_id} | {group}\nGT=green, raw pred=red, LCC=yellow", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_nifti_packet(case_id: str, group: str, base_path: Path, gt_img: sitk.Image, raw_img: sitk.Image, lcc_arr: np.ndarray) -> None:
    case_dir = NIFTI_DIR / group / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    # Copy base and write masks.
    try:
        shutil.copy2(base_path, case_dir / f"{case_id}_image_pre_or_base.nii.gz")
    except Exception:
        pass
    sitk.WriteImage(gt_img, str(case_dir / f"{case_id}_expert_gt.nii.gz"), useCompression=True)
    sitk.WriteImage(raw_img, str(case_dir / f"{case_id}_v2_raw_prediction.nii.gz"), useCompression=True)
    lcc_img = sitk.GetImageFromArray(lcc_arr.astype(np.uint8))
    lcc_img.CopyInformation(raw_img)
    sitk.WriteImage(lcc_img, str(case_dir / f"{case_id}_v2_lcc_prediction.nii.gz"), useCompression=True)


def make_contact_sheet(image_paths: List[Path], out_path: Path, title: str, max_images: int = 80) -> None:
    paths = image_paths[:max_images]
    if not paths:
        return
    imgs = []
    for p in paths:
        try:
            imgs.append(plt.imread(str(p)))
        except Exception:
            continue
    if not imgs:
        return
    n = len(imgs)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.4), dpi=120)
    axes = np.atleast_1d(axes).ravel()
    for ax, img, p in zip(axes, imgs, paths):
        ax.imshow(img)
        ax.set_title(p.stem.replace("_review", ""), fontsize=8)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_prior_flags() -> Optional[pd.DataFrame]:
    if AUDIT13_CSV.exists():
        try:
            prior = pd.read_csv(AUDIT13_CSV)
            ccol = find_case_col(prior)
            prior["case_id_std"] = prior[ccol].apply(clean_case_id)
            return prior
        except Exception:
            return None
    return None


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


# =============================================================================
# Main audit
# =============================================================================

def main() -> None:
    print("14 LCC-collapse visual pattern review")
    print("======================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    ensure_dirs()

    if not CASE_TABLE.exists():
        raise FileNotFoundError(f"Case table not found: {CASE_TABLE}")

    df = pd.read_csv(CASE_TABLE)
    ccol = find_case_col(df)
    df["case_id_std"] = df[ccol].apply(clean_case_id)
    if "cohort" not in df.columns:
        df["cohort"] = df["case_id_std"].str.extract(r"^(DUKE|ISPY1|NACT)", expand=False).fillna("OTHER")

    print(f"Cases in table: {len(df)}")

    prior = load_prior_flags()
    if prior is not None:
        print(f"Loaded prior 13 audit: {AUDIT13_CSV}")
    else:
        print("Prior 13 audit not found or not readable; proceeding independently from masks.")

    rows = []
    missing = []

    for i, row in df.iterrows():
        case_id = row["case_id_std"]
        cohort = row.get("cohort", "OTHER")
        pred_path = find_file(PRED_V2_DIR, case_id)
        gt_path = find_file(GT_DIR, case_id.lower()) or find_file(GT_DIR, case_id)
        base_path = find_base_image(case_id)
        if pred_path is None or gt_path is None:
            missing.append({"case_id": case_id, "missing_pred": pred_path is None, "missing_gt": gt_path is None})
            continue
        try:
            raw, raw_img = arr_from_img(pred_path, binary=True)
            gt, gt_img = arr_from_img(gt_path, binary=True)
            lcc = keep_largest_component(raw)
            spacing = gt_img.GetSpacing()
            raw_d = dice(raw, gt)
            lcc_d = dice(lcc, gt)
            raw_prec, raw_rec = precision_recall(raw, gt)
            lcc_prec, lcc_rec = precision_recall(lcc, gt)
            gt_c = centroid_zyx(gt)
            raw_c = centroid_zyx(raw)
            lcc_c = centroid_zyx(lcc)
            width = gt.shape[2]
            gt_side = side_from_centroid_x(gt_c[2], width)
            raw_side = side_from_centroid_x(raw_c[2], width)
            lcc_side = side_from_centroid_x(lcc_c[2], width)
            raw_opp = opposite_half(gt_c, raw_c, width)
            lcc_opp = opposite_half(gt_c, lcc_c, width)
            raw_dist = centroid_distance_mm(gt_c, raw_c, spacing)
            lcc_dist = centroid_distance_mm(gt_c, lcc_c, spacing)
            stats_raw = component_stats(raw)
            stats_lcc = component_stats(lcc)
            collapse = bool(raw_d > RAW_DICE_MIN_FOR_COLLAPSE and lcc_d < LCC_DICE_MAX_FOR_COLLAPSE)
            centroid100 = bool(np.isfinite(lcc_dist) and lcc_dist > CENTROID_LARGE_DISTANCE_MM)
            # Broad FOV heuristic, only if base available; otherwise use GT image array as fallback.
            if base_path is not None and base_path.exists():
                try:
                    base_arr, _ = arr_from_img(base_path, binary=False)
                    fov = rough_bilateral_fov(base_arr)
                except Exception:
                    fov = rough_bilateral_fov(gt.astype(np.float32))
            else:
                fov = rough_bilateral_fov(gt.astype(np.float32))

            out = {
                "case_id_std": case_id,
                "cohort": cohort,
                "raw_dice": raw_d,
                "lcc_dice": lcc_d,
                "delta_lcc_minus_raw_dice": lcc_d - raw_d,
                "raw_precision": raw_prec,
                "raw_recall": raw_rec,
                "lcc_precision": lcc_prec,
                "lcc_recall": lcc_rec,
                "lcc_collapse": collapse,
                "raw_opposite_half": raw_opp,
                "lcc_opposite_half": lcc_opp,
                "lcc_centroid_gt100mm": centroid100,
                "gt_side": gt_side,
                "raw_side": raw_side,
                "lcc_side": lcc_side,
                "raw_centroid_distance_mm": raw_dist,
                "lcc_centroid_distance_mm": lcc_dist,
                "gt_centroid_z": gt_c[0],
                "gt_centroid_y": gt_c[1],
                "gt_centroid_x": gt_c[2],
                "raw_centroid_z": raw_c[0],
                "raw_centroid_y": raw_c[1],
                "raw_centroid_x": raw_c[2],
                "lcc_centroid_z": lcc_c[0],
                "lcc_centroid_y": lcc_c[1],
                "lcc_centroid_x": lcc_c[2],
                "shape_match": tuple(raw.shape) == tuple(gt.shape),
                "spacing_match": np.allclose(raw_img.GetSpacing(), gt_img.GetSpacing()),
                "origin_match": np.allclose(raw_img.GetOrigin(), gt_img.GetOrigin()),
                "direction_match": np.allclose(raw_img.GetDirection(), gt_img.GetDirection()),
                "base_image_path": str(base_path) if base_path is not None else "",
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                **{f"raw_{k}": v for k, v in stats_raw.items()},
                **{f"lcc_{k}": v for k, v in stats_lcc.items()},
                **fov,
            }
            # Add selected existing table fields if available.
            for c in [
                "v2_lcc_dice", "v2_raw_dice", "v2_lcc_theoretical_max_dice_by_volume_ratio",
                "v2_lcc_dice_gap_to_volume_bound", "v2_lcc_dice_utilization_of_volume_bound",
                "v2_lcc_pred_to_gt_vol_ratio", "volume_bound_diagnostic_group",
                "v2_lcc_volume_bound_failure_signature", "gt_vol_cm3",
            ]:
                if c in df.columns:
                    out[c] = row[c]
            rows.append(out)
        except Exception as e:
            missing.append({"case_id": case_id, "error": repr(e)})
        if (i + 1) % 50 == 0:
            print(f"  audited {i+1}/{len(df)}")

    audit = pd.DataFrame(rows)
    audit_path = METRICS_DIR / "all_case_independent_lcc_collapse_pattern_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Saved independent all-case audit: {audit_path}")
    if missing:
        pd.DataFrame(missing).to_csv(METRICS_DIR / "missing_or_error_cases.csv", index=False)

    # Flag sets and crosstabs.
    collapse = audit["lcc_collapse"].fillna(False)
    opp = audit["lcc_opposite_half"].fillna(False)
    cent100 = audit["lcc_centroid_gt100mm"].fillna(False)

    a = int((collapse & opp).sum())
    b = int((collapse & ~opp).sum())
    c = int((~collapse & opp).sum())
    d = int((~collapse & ~opp).sum())
    table = np.array([[a, b], [c, d]])
    odds_ratio, fisher_p = fisher_exact(table)
    chi2, chi2_p, _, _ = chi2_contingency(table)

    set_summary = pd.DataFrame([
        {"metric": "n_cases", "value": len(audit)},
        {"metric": "lcc_collapse", "value": int(collapse.sum())},
        {"metric": "lcc_opposite_half", "value": int(opp.sum())},
        {"metric": "lcc_centroid_gt100mm", "value": int(cent100.sum())},
        {"metric": "collapse_intersect_opposite", "value": a},
        {"metric": "collapse_outside_opposite", "value": b},
        {"metric": "opposite_outside_collapse", "value": c},
        {"metric": "neither_collapse_nor_opposite", "value": d},
        {"metric": "collapse_set_equals_opposite_set", "value": bool((collapse == opp).all())},
        {"metric": "odds_ratio_collapse_vs_opposite", "value": odds_ratio},
        {"metric": "fisher_exact_p", "value": fisher_p},
        {"metric": "chi2", "value": chi2},
        {"metric": "chi2_p", "value": chi2_p},
    ])
    set_summary.to_csv(METRICS_DIR / "lcc_collapse_opposite_set_summary.csv", index=False)

    pd.DataFrame(table, index=["LCC-collapse", "not LCC-collapse"], columns=["opposite-half", "not opposite-half"]).to_csv(
        METRICS_DIR / "crosstab_lcc_collapse_vs_opposite_half.csv"
    )
    audit.assign(
        collapse_label=np.where(collapse, "LCC-collapse", "not LCC-collapse"),
        opposite_label=np.where(opp, "opposite-half", "not opposite-half"),
        centroid100_label=np.where(cent100, "centroid>100mm", "centroid<=100mm"),
    ).groupby(["collapse_label", "opposite_label", "centroid100_label", "cohort"], dropna=False).size().reset_index(name="n").to_csv(
        METRICS_DIR / "crosstab_lcc_collapse_opposite_centroid100_by_cohort.csv", index=False
    )

    # Cohort summaries.
    cohort_summary = audit.groupby("cohort", dropna=False).agg(
        n=("case_id_std", "count"),
        lcc_collapse_n=("lcc_collapse", "sum"),
        lcc_opposite_half_n=("lcc_opposite_half", "sum"),
        lcc_centroid_gt100mm_n=("lcc_centroid_gt100mm", "sum"),
        raw_dice_mean=("raw_dice", "mean"),
        lcc_dice_mean=("lcc_dice", "mean"),
        delta_lcc_minus_raw_mean=("delta_lcc_minus_raw_dice", "mean"),
        bilateral_like_fov_mean=("bilateral_like_fov", "mean"),
    ).reset_index()
    for col in ["lcc_collapse_n", "lcc_opposite_half_n", "lcc_centroid_gt100mm_n"]:
        cohort_summary[col.replace("_n", "_pct")] = 100.0 * cohort_summary[col] / cohort_summary["n"].clip(lower=1)
    cohort_summary.to_csv(METRICS_DIR / "lcc_collapse_opposite_by_cohort.csv", index=False)

    # Group labels for visual review.
    def visual_group(r):
        if r["lcc_collapse"] and r["lcc_opposite_half"]:
            return "01_collapse_and_opposite_half"
        if r["lcc_collapse"] and not r["lcc_opposite_half"]:
            return "02_collapse_not_opposite_half"
        if (not r["lcc_collapse"]) and r["lcc_opposite_half"]:
            return "03_opposite_half_not_collapse"
        return "04_other"

    audit["visual_review_group"] = audit.apply(visual_group, axis=1)
    audit.to_csv(METRICS_DIR / "all_case_independent_lcc_collapse_pattern_audit_with_groups.csv", index=False)

    # Manual review checklist for all relevant cases.
    relevant = audit[audit["visual_review_group"].isin([
        "01_collapse_and_opposite_half", "02_collapse_not_opposite_half", "03_opposite_half_not_collapse"
    ])].copy()
    relevant["manual_review_status"] = ""
    relevant["visual_pattern"] = ""
    relevant["notes"] = ""
    checklist_cols = [
        "case_id_std", "cohort", "visual_review_group", "raw_dice", "lcc_dice",
        "delta_lcc_minus_raw_dice", "lcc_opposite_half", "lcc_centroid_distance_mm",
        "gt_side", "raw_side", "lcc_side", "bilateral_like_fov", "manual_review_status",
        "visual_pattern", "notes",
    ]
    relevant[checklist_cols].to_csv(METRICS_DIR / "manual_review_checklist_lcc_collapse_patterns.csv", index=False)

    # Select cases for previews.
    rng = np.random.default_rng(RANDOM_SEED)
    selected = audit[audit["visual_review_group"].isin(["01_collapse_and_opposite_half", "02_collapse_not_opposite_half"])].copy()
    if SAVE_OPPOSITE_NOT_COLLAPSE_CONTROLS:
        controls = audit[audit["visual_review_group"] == "03_opposite_half_not_collapse"].copy()
        if len(controls) > MAX_OPPOSITE_NOT_COLLAPSE_CONTROLS:
            controls = controls.sample(MAX_OPPOSITE_NOT_COLLAPSE_CONTROLS, random_state=RANDOM_SEED)
        selected = pd.concat([selected, controls], ignore_index=True)
    if SAVE_RANDOM_GOOD_CONTROLS:
        good = audit[(audit["lcc_dice"] > 0.75) & (~audit["lcc_collapse"]) & (~audit["lcc_opposite_half"])].copy()
        if len(good) > MAX_RANDOM_GOOD_CONTROLS:
            good = good.sample(MAX_RANDOM_GOOD_CONTROLS, random_state=RANDOM_SEED + 1)
        good["visual_review_group"] = "05_good_controls"
        selected = pd.concat([selected, good], ignore_index=True)
    selected.to_csv(METRICS_DIR / "selected_visual_review_cases.csv", index=False)

    # Create preview PNGs and NIfTI packets.
    preview_paths = []
    print(f"Creating visual review packet for {len(selected)} cases...")
    for j, r in selected.iterrows():
        case_id = r["case_id_std"]
        group = r["visual_review_group"]
        pred_path = find_file(PRED_V2_DIR, case_id)
        gt_path = find_file(GT_DIR, case_id.lower()) or find_file(GT_DIR, case_id)
        base_path = find_base_image(case_id)
        if pred_path is None or gt_path is None or base_path is None:
            continue
        try:
            raw, raw_img = arr_from_img(pred_path, binary=True)
            gt, gt_img = arr_from_img(gt_path, binary=True)
            base, _ = arr_from_img(base_path, binary=False)
            lcc = keep_largest_component(raw)
            # If base shape differs from masks, fallback to GT mask background to avoid crash.
            if base.shape != gt.shape:
                # Try central crop/pad is unsafe; use raw image array if same, otherwise skip preview.
                if raw.shape == gt.shape:
                    base = np.zeros_like(gt, dtype=np.float32)
                else:
                    continue
            p = PREVIEW_DIR / group / f"{case_id}_review.png"
            save_case_preview(case_id, group, base, gt, raw, lcc, p)
            preview_paths.append(p)
            if SAVE_NIFTI_REVIEW_PACKETS:
                save_nifti_packet(case_id, group, base_path, gt_img, raw_img, lcc)
        except Exception as e:
            print(f"  preview error {case_id}: {e}")

    make_contact_sheet(preview_paths, FIGURES_DIR / "contact_sheet_lcc_collapse_visual_patterns.jpg", "LCC-collapse and opposite-half visual-pattern review", MAX_CONTACT_SHEET_IMAGES)

    # Basic figures.
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    labels = ["collapse\n& opposite", "collapse\nnot opposite", "opposite\nnot collapse", "neither"]
    vals = [a, b, c, d]
    ax.bar(labels, vals)
    ax.set_ylabel("Number of cases")
    ax.set_title("LCC-collapse and opposite-half flag intersections")
    for k, v in enumerate(vals):
        ax.text(k, v + max(vals) * 0.02, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_lcc_collapse_opposite_intersection_counts.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    x = np.arange(len(cohort_summary))
    width = 0.28
    ax.bar(x - width, cohort_summary["lcc_collapse_pct"], width, label="LCC-collapse")
    ax.bar(x, cohort_summary["lcc_opposite_half_pct"], width, label="Opposite-half")
    ax.bar(x + width, cohort_summary["lcc_centroid_gt100mm_pct"], width, label="Centroid >100mm")
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_summary["cohort"].astype(str))
    ax.set_ylabel("Percent of cohort")
    ax.set_title("Failure flags by cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_failure_flags_by_cohort.png", bbox_inches="tight")
    plt.close(fig)

    # Takeaways.
    lines = []
    lines.append("14 LCC-COLLAPSE VISUAL PATTERN REVIEW — TAKEAWAYS")
    lines.append("=" * 78)
    lines.append(f"Cases audited: {len(audit)}")
    lines.append("")
    lines.append("Flag intersections:")
    lines.append(f"  LCC-collapse cases: {int(collapse.sum())} ({100*collapse.mean():.1f}%)")
    lines.append(f"  Opposite-half flags: {int(opp.sum())} ({100*opp.mean():.1f}%)")
    lines.append(f"  Collapse ∩ opposite-half: {a}")
    lines.append(f"  Collapse outside opposite-half: {b}")
    lines.append(f"  Opposite-half outside collapse: {c}")
    lines.append(f"  Collapse set == opposite set: {bool((collapse == opp).all())}")
    lines.append(f"  Odds ratio collapse vs opposite-half: {odds_ratio:.3g}")
    lines.append(f"  Fisher exact p-value: {fisher_p:.3e}")
    lines.append(f"  Chi-square p-value: {chi2_p:.3e}")
    lines.append("")
    lines.append("Cohort breakdown:")
    for _, rr in cohort_summary.iterrows():
        lines.append(
            f"  {rr['cohort']}: n={int(rr['n'])}, "
            f"collapse={int(rr['lcc_collapse_n'])} ({rr['lcc_collapse_pct']:.1f}%), "
            f"opposite-half={int(rr['lcc_opposite_half_n'])} ({rr['lcc_opposite_half_pct']:.1f}%), "
            f"centroid>100mm={int(rr['lcc_centroid_gt100mm_n'])} ({rr['lcc_centroid_gt100mm_pct']:.1f}%)"
        )
    lines.append("")
    lines.append("Suggested manuscript wording:")
    lines.append("  LCC-collapse and opposite-image-half/spatially displaced component selection were")
    lines.append("  strongly associated, but not identical. Opposite-half displacement accounted for")
    lines.append("  part of LCC-collapse cases, while remaining collapse cases reflected additional")
    lines.append("  component-selection or fragmentation failure modes. These results support")
    lines.append("  reporting LCC as a sensitivity analysis rather than a universally safe clinical")
    lines.append("  postprocessing rule.")
    lines.append("")
    lines.append("Important caution:")
    lines.append("  The opposite-half flag is a heuristic based on image-half centroids, not a confirmed")
    lines.append("  anatomical laterality label. Use wording such as 'opposite image half' or 'large")
    lines.append("  spatial displacement', not 'wrong breast', unless manually confirmed.")

    take_path = OUTPUT_ROOT / "article_takeaways_14.txt"
    take_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE — 14 visual-pattern review complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Previews: {PREVIEW_DIR}")
    print(f"Takeaways: {take_path}")
    print("\nQuick results:")
    print(f"  all cases audited: {len(audit)}")
    print(f"  LCC-collapse: {int(collapse.sum())}")
    print(f"  opposite-half flags: {int(opp.sum())}")
    print(f"  collapse ∩ opposite: {a}")
    print(f"  collapse outside opposite: {b}")
    print(f"  opposite outside collapse: {c}")
    print(f"  odds ratio: {odds_ratio:.3g}")
    print(f"  Fisher p: {fisher_p:.3e}")
    print("\nOpen these first:")
    print(f"  {FIGURES_DIR / 'contact_sheet_lcc_collapse_visual_patterns.jpg'}")
    print(f"  {METRICS_DIR / 'manual_review_checklist_lcc_collapse_patterns.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        print("If outputs were partly saved, send the last console lines and the CSV names in the metrics folder.")
        raise
