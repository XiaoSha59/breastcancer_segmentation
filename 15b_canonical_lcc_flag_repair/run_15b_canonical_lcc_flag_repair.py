# run_15b_canonical_lcc_flag_repair.py
# Canonical LCC-collapse / opposite-image-half audit and corrected review packet.
#
# Purpose:
#   - Stop using inconsistent opposite-half flags from previous scripts.
#   - Recompute LCC-collapse and opposite-image-half flags once, deterministically, from masks.
#   - Compare previous D13/D14/D15 flag sets against this canonical result.
#   - Rebuild the visual review packet from the exact canonical union:
#       LCC-collapse OR opposite-image-half.
#
# Run from:
#   C:\Users\olegk\Desktop\MRI Project\15_final_lcc_collapse_profile
#
# Command:
#   python run_15b_canonical_lcc_flag_repair.py

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except Exception as e:
    raise RuntimeError("SimpleITK is required. Install with: pip install SimpleITK") from e

try:
    from scipy import ndimage
    from scipy.stats import fisher_exact, chi2_contingency
except Exception:
    ndimage = None
    fisher_exact = None
    chi2_contingency = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from e


# =============================================================================
# CONFIG — edit only if your folders are different
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")

# External predictions and masks
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"

# Optional image background for previews: nnU-Net prepared channel 0000 is easiest
INPUT_V2_DIR = Path(r"C:\nnw\mama_mia_input_v2")
RAW_IMAGES_DIR = PROJECT_ROOT / "images_2"

# Source all-case table: used only for case list and extra metadata, not for flags.
D10_CASE_TABLE = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "final_sanity_case_table_with_volume_bounds.csv"

# Previous audit tables to compare against. Missing files are okay.
D13_TABLE = PROJECT_ROOT / "13_final_lcc_side_audit" / "metrics" / "independent_lcc_side_audit_all_cases.csv"
D14_TABLE = PROJECT_ROOT / "14_lcc_collapse_visual_pattern_review" / "metrics" / "all_case_independent_lcc_collapse_pattern_audit.csv"
D15_TABLE = PROJECT_ROOT / "15_final_lcc_collapse_profile" / "metrics" / "all_case_lcc_collapse_profile_table.csv"

# Output
OUTPUT_ROOT = PROJECT_ROOT / "15_final_lcc_collapse_profile" / "15b_canonical_lcc_flag_repair"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURES_DIR = OUTPUT_ROOT / "figures"
PREVIEW_DIR = OUTPUT_ROOT / "review_packet" / "previews"
NIFTI_DIR = OUTPUT_ROOT / "review_packet" / "nifti_cases"

# Canonical flag definitions
RAW_DICE_COLLAPSE_MIN = 0.20
LCC_DICE_COLLAPSE_MAX = 0.05
OPPOSITE_MIN_X_DISTANCE_MM = 30.0
LARGE_CENTROID_DISTANCE_MM = 100.0

# Preview controls
SAVE_NIFTI_PACKET = True
MAX_CONTACT_SHEET_CASES = 120  # if union is larger, individual PNGs still saved; contact sheet caps


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    for p in [OUTPUT_ROOT, METRICS_DIR, FIGURES_DIR, PREVIEW_DIR, NIFTI_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def normalize_case_id(case_id: str) -> str:
    s = str(case_id).strip()
    s = s.replace(".nii.gz", "").replace(".nii", "")
    return s.upper()


def cohort_from_case(case_id: str) -> str:
    s = normalize_case_id(case_id)
    if s.startswith("DUKE"):
        return "DUKE"
    if s.startswith("ISPY1"):
        return "ISPY1"
    if s.startswith("NACT"):
        return "NACT"
    return "OTHER"


def find_case_col(df: pd.DataFrame) -> str:
    candidates = [
        "case_id_std", "case_id_upper", "case_id", "pid", "patient_id", "id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No case-id column found. Available columns: {list(df.columns)[:30]}")


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def find_flag_col(df: pd.DataFrame, kind: str) -> Optional[str]:
    """Find likely previous flag columns. Conservative and prints what it used."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    if kind == "collapse":
        exact = [
            "lcc_collapse", "is_lcc_collapse", "v2_lcc_collapse",
            "lcc_collapse_flag", "collapse_flag",
            "canonical_lcc_collapse",
        ]
        for name in exact:
            if name in lower:
                return lower[name]
        # heuristic: contains lcc and collapse, not summary text
        cand = [c for c in cols if ("lcc" in c.lower() and "collapse" in c.lower())]
        return cand[0] if cand else None

    if kind == "opposite":
        exact = [
            "opposite_half", "opposite_image_half", "opposite_half_flag",
            "lcc_opposite_half", "lcc_opposite_half_flag",
            "opposite_side", "opposite_side_flag", "lcc_opposite_side_flag",
            "lcc_opposite_image_half",
        ]
        for name in exact:
            if name in lower:
                return lower[name]
        cand = [c for c in cols if ("opposite" in c.lower() and ("half" in c.lower() or "side" in c.lower()))]
        return cand[0] if cand else None

    if kind == "centroid100":
        exact = [
            "large_centroid_distance_gt_100mm", "centroid_distance_gt_100mm",
            "lcc_centroid_distance_gt_100mm", "large_lcc_centroid_distance_gt_100mm"
        ]
        for name in exact:
            if name in lower:
                return lower[name]
        cand = [c for c in cols if ("centroid" in c.lower() and "100" in c.lower())]
        return cand[0] if cand else None

    return None


def load_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def img_to_arr(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def binary_arr(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = load_sitk(path)
    arr = img_to_arr(img)
    return (arr > 0).astype(np.uint8), img


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    if denom > 0 and (pred_b.sum() == 0 or gt_b.sum() == 0):
        return 0.0
    return float(2.0 * np.logical_and(pred_b, gt_b).sum() / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    prec = float(tp / pred_sum) if pred_sum > 0 else 0.0
    rec = float(tp / gt_sum) if gt_sum > 0 else 0.0
    return prec, rec


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    if ndimage is None:
        raise RuntimeError("scipy is required for connected components. Install with: pip install scipy")
    labeled, n = ndimage.label(mask.astype(bool))
    if n <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return (labeled == largest).astype(np.uint8)


def component_stats(mask: np.ndarray) -> Dict[str, float]:
    out = {
        "n_components": 0,
        "largest_component_voxels": 0.0,
        "second_largest_component_voxels": 0.0,
        "largest_component_fraction": np.nan,
        "second_largest_component_fraction": np.nan,
    }
    total = int(mask.sum())
    if total == 0:
        return out
    if ndimage is None:
        return out
    labeled, n = ndimage.label(mask.astype(bool))
    out["n_components"] = int(n)
    if n <= 0:
        return out
    counts = np.bincount(labeled.ravel())
    counts = counts[1:]  # skip background
    counts_sorted = np.sort(counts)[::-1]
    largest = float(counts_sorted[0]) if len(counts_sorted) else 0.0
    second = float(counts_sorted[1]) if len(counts_sorted) > 1 else 0.0
    out["largest_component_voxels"] = largest
    out["second_largest_component_voxels"] = second
    out["largest_component_fraction"] = largest / total if total else np.nan
    out["second_largest_component_fraction"] = second / total if total else np.nan
    return out


def centroid_index(mask: np.ndarray) -> Optional[np.ndarray]:
    pts = np.argwhere(mask.astype(bool))
    if pts.size == 0:
        return None
    return pts.mean(axis=0)  # z, y, x


def centroid_mm_distance(c1: Optional[np.ndarray], c2: Optional[np.ndarray], spacing_xyz: Tuple[float, float, float]) -> float:
    if c1 is None or c2 is None:
        return np.nan
    # c is z,y,x. spacing from ITK is x,y,z.
    dz = (c1[0] - c2[0]) * spacing_xyz[2]
    dy = (c1[1] - c2[1]) * spacing_xyz[1]
    dx = (c1[2] - c2[2]) * spacing_xyz[0]
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def centroid_x_mm_distance(c1: Optional[np.ndarray], c2: Optional[np.ndarray], spacing_xyz: Tuple[float, float, float]) -> float:
    if c1 is None or c2 is None:
        return np.nan
    return float(abs(c1[2] - c2[2]) * spacing_xyz[0])


def opposite_image_half(c_gt: Optional[np.ndarray], c_pred: Optional[np.ndarray], shape: Tuple[int, int, int], spacing_xyz: Tuple[float, float, float]) -> bool:
    if c_gt is None or c_pred is None:
        return False
    mid_x = (shape[2] - 1) / 2.0
    # Require centroids to be on opposite image halves and separated enough along x.
    opposite = (c_gt[2] - mid_x) * (c_pred[2] - mid_x) < 0
    x_dist_mm = abs(c_gt[2] - c_pred[2]) * spacing_xyz[0]
    return bool(opposite and x_dist_mm >= OPPOSITE_MIN_X_DISTANCE_MM)


def spacing_tuple(img: sitk.Image) -> Tuple[float, float, float]:
    sp = img.GetSpacing()
    if len(sp) >= 3:
        return float(sp[0]), float(sp[1]), float(sp[2])
    return float(sp[0]), float(sp[1]), 1.0


def geometry_info(img: sitk.Image) -> Dict[str, str]:
    return {
        "spacing": str(tuple(round(float(x), 5) for x in img.GetSpacing())),
        "origin": str(tuple(round(float(x), 5) for x in img.GetOrigin())),
        "direction": str(tuple(round(float(x), 5) for x in img.GetDirection())),
        "size": str(tuple(int(x) for x in img.GetSize())),
    }


def geometry_mismatch(img_a: sitk.Image, img_b: sitk.Image, tol: float = 1e-4) -> Dict[str, bool]:
    def arr_close(x, y):
        return np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), atol=tol, rtol=0)
    return {
        "size_mismatch": tuple(img_a.GetSize()) != tuple(img_b.GetSize()),
        "spacing_mismatch": not arr_close(img_a.GetSpacing(), img_b.GetSpacing()),
        "origin_mismatch": not arr_close(img_a.GetOrigin(), img_b.GetOrigin()),
        "direction_mismatch": not arr_close(img_a.GetDirection(), img_b.GetDirection()),
    }


def find_pred_path(case_id: str, pred_dir: Path) -> Optional[Path]:
    cid = normalize_case_id(case_id)
    candidates = [
        pred_dir / f"{cid}.nii.gz",
        pred_dir / f"{cid.lower()}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    # last resort glob
    hits = list(pred_dir.glob(f"{cid}*.nii.gz")) + list(pred_dir.glob(f"{cid.lower()}*.nii.gz"))
    return hits[0] if hits else None


def find_gt_path(case_id: str, gt_dir: Path) -> Optional[Path]:
    cid = normalize_case_id(case_id)
    candidates = [
        gt_dir / f"{cid.lower()}.nii.gz",
        gt_dir / f"{cid}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(gt_dir.glob(f"{cid.lower()}*.nii.gz")) + list(gt_dir.glob(f"{cid}*.nii.gz"))
    return hits[0] if hits else None


def find_background_image(case_id: str) -> Optional[Path]:
    cid = normalize_case_id(case_id)
    # Prefer prepared nnU-Net v2 input channel 0000 because it matches inference input.
    p = INPUT_V2_DIR / f"{cid}_0000.nii.gz"
    if p.exists():
        return p
    p = INPUT_V2_DIR / f"{cid.lower()}_0000.nii.gz"
    if p.exists():
        return p
    # Fallback raw phase 0
    case_dir = RAW_IMAGES_DIR / cid
    if not case_dir.exists():
        case_dir = RAW_IMAGES_DIR / cid.lower()
    if case_dir.exists():
        files = sorted(case_dir.glob("*.nii.gz"))
        if files:
            return files[0]
    return None


def intensity_window(slice_arr: np.ndarray) -> Tuple[float, float]:
    finite = slice_arr[np.isfinite(slice_arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def choose_slice(gt: np.ndarray, raw: np.ndarray, lcc: np.ndarray) -> int:
    combined = (gt > 0).astype(np.uint8) + (raw > 0).astype(np.uint8) + (lcc > 0).astype(np.uint8)
    sums = combined.reshape(combined.shape[0], -1).sum(axis=1)
    if sums.max() > 0:
        return int(np.argmax(sums))
    return int(combined.shape[0] // 2)


def overlay_contours(ax, mask: np.ndarray, z: int, color: str, linewidth: float = 1.0):
    sl = mask[z].astype(float)
    if sl.max() <= 0:
        return
    try:
        ax.contour(sl, levels=[0.5], colors=[color], linewidths=linewidth)
    except Exception:
        pass


def make_preview(case_id: str, bg_arr: np.ndarray, gt: np.ndarray, raw: np.ndarray, lcc: np.ndarray, row: Dict[str, object], out_path: Path) -> None:
    z = choose_slice(gt, raw, lcc)
    bg_slice = bg_arr[z] if bg_arr is not None and bg_arr.ndim == 3 and z < bg_arr.shape[0] else np.zeros_like(gt[z], dtype=float)
    lo, hi = intensity_window(bg_slice)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    title = (
        f"{case_id} | rawD={row.get('canonical_v2_raw_dice', np.nan):.3f} "
        f"LCCD={row.get('canonical_v2_lcc_dice', np.nan):.3f} "
        f"opp={row.get('canonical_lcc_opposite_half', False)} "
        f"collapse={row.get('canonical_lcc_collapse', False)}"
    )
    fig.suptitle(title, fontsize=9)

    panels = [
        ("GT only", ["gt"]),
        ("GT + raw", ["gt", "raw"]),
        ("GT + LCC", ["gt", "lcc"]),
        ("Raw + LCC", ["raw", "lcc"]),
    ]
    for ax, (name, layers) in zip(axes, panels):
        ax.imshow(bg_slice, cmap="gray", vmin=lo, vmax=hi)
        if "gt" in layers:
            overlay_contours(ax, gt, z, "lime", 1.2)
        if "raw" in layers:
            overlay_contours(ax, raw, z, "red", 1.0)
        if "lcc" in layers:
            overlay_contours(ax, lcc, z, "yellow", 1.0)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(image_paths: Sequence[Path], out_path: Path, title: str, max_images: int = MAX_CONTACT_SHEET_CASES) -> None:
    paths = list(image_paths)[:max_images]
    if not paths:
        return
    import matplotlib.image as mpimg

    n = len(paths)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig_w = 14
    fig_h = max(4, rows * 3.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_arr = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    fig.suptitle(title, fontsize=12)
    for ax, p in zip(axes_arr, paths):
        try:
            img = mpimg.imread(str(p))
            ax.imshow(img)
            ax.set_title(p.stem.replace("_canonical_preview", ""), fontsize=7)
            ax.axis("off")
        except Exception:
            ax.axis("off")
    for ax in axes_arr[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def odds_ratio_from_counts(a: int, b: int, c: int, d: int) -> float:
    # Haldane correction if any cell is zero
    aa, bb, cc, dd = map(float, [a, b, c, d])
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5; bb += 0.5; cc += 0.5; dd += 0.5
    return float((aa * dd) / (bb * cc))


def compare_previous_sets(canonical: pd.DataFrame, previous_path: Path, label: str) -> Dict[str, object]:
    out = {"source": label, "path": str(previous_path), "status": "missing"}
    if not previous_path.exists():
        return out

    try:
        prev = pd.read_csv(previous_path)
        case_col = find_case_col(prev)
        prev["case_id_std"] = prev[case_col].apply(normalize_case_id)

        c_col = find_flag_col(prev, "collapse")
        o_col = find_flag_col(prev, "opposite")
        out["status"] = "loaded"
        out["collapse_col_used"] = c_col
        out["opposite_col_used"] = o_col

        can_collapse = set(canonical.loc[canonical["canonical_lcc_collapse"], "case_id_std"])
        can_opp = set(canonical.loc[canonical["canonical_lcc_opposite_half"], "case_id_std"])

        if c_col is not None:
            prev_collapse = set(prev.loc[as_bool_series(prev[c_col]), "case_id_std"])
            out["prev_collapse_n"] = len(prev_collapse)
            out["canonical_collapse_n"] = len(can_collapse)
            out["collapse_intersection"] = len(prev_collapse & can_collapse)
            out["collapse_prev_minus_canonical"] = len(prev_collapse - can_collapse)
            out["collapse_canonical_minus_prev"] = len(can_collapse - prev_collapse)
            pd.DataFrame({"case_id_std": sorted(prev_collapse - can_collapse)}).to_csv(
                METRICS_DIR / f"{label}_collapse_prev_minus_canonical.csv", index=False
            )
            pd.DataFrame({"case_id_std": sorted(can_collapse - prev_collapse)}).to_csv(
                METRICS_DIR / f"{label}_collapse_canonical_minus_prev.csv", index=False
            )

        if o_col is not None:
            prev_opp = set(prev.loc[as_bool_series(prev[o_col]), "case_id_std"])
            out["prev_opposite_n"] = len(prev_opp)
            out["canonical_opposite_n"] = len(can_opp)
            out["opposite_intersection"] = len(prev_opp & can_opp)
            out["opposite_prev_minus_canonical"] = len(prev_opp - can_opp)
            out["opposite_canonical_minus_prev"] = len(can_opp - prev_opp)
            pd.DataFrame({"case_id_std": sorted(prev_opp - can_opp)}).to_csv(
                METRICS_DIR / f"{label}_opposite_prev_minus_canonical.csv", index=False
            )
            pd.DataFrame({"case_id_std": sorted(can_opp - prev_opp)}).to_csv(
                METRICS_DIR / f"{label}_opposite_canonical_minus_prev.csv", index=False
            )

        return out
    except Exception as e:
        out["status"] = f"error: {repr(e)}"
        return out


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(group_col, dropna=False):
        rows.append({
            group_col: val,
            "n": len(g),
            "dice_raw_mean": g["canonical_v2_raw_dice"].mean(),
            "dice_lcc_mean": g["canonical_v2_lcc_dice"].mean(),
            "precision_lcc_mean": g["canonical_v2_lcc_precision"].mean(),
            "recall_lcc_mean": g["canonical_v2_lcc_recall"].mean(),
            "centroid_distance_lcc_mm_mean": g["canonical_lcc_centroid_distance_mm"].mean(),
            "centroid_x_distance_lcc_mm_mean": g["canonical_lcc_centroid_x_distance_mm"].mean(),
            "raw_n_components_mean": g["canonical_raw_n_components"].mean(),
            "raw_largest_component_fraction_mean": g["canonical_raw_largest_component_fraction"].mean(),
            "lcc_pred_voxels_mean": g["canonical_lcc_voxels"].mean(),
            "gt_voxels_mean": g["canonical_gt_voxels"].mean(),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# =============================================================================
# Main canonical audit
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("15b canonical LCC/opposite-half repair")
    print("======================================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Case table   : {D10_CASE_TABLE}")
    print()

    if not D10_CASE_TABLE.exists():
        raise FileNotFoundError(f"Could not find case table: {D10_CASE_TABLE}")

    base = pd.read_csv(D10_CASE_TABLE)
    case_col = find_case_col(base)
    base["case_id_std"] = base[case_col].apply(normalize_case_id)
    case_ids = sorted(base["case_id_std"].unique())
    print(f"Cases from D10 table: {len(case_ids)}")

    rows = []
    preview_rows = []
    preview_paths = []

    for i, case_id in enumerate(case_ids, 1):
        pred_path = find_pred_path(case_id, PRED_V2_DIR)
        gt_path = find_gt_path(case_id, GT_DIR)
        bg_path = find_background_image(case_id)

        row = {
            "case_id_std": case_id,
            "cohort": cohort_from_case(case_id),
            "pred_path": str(pred_path) if pred_path else "",
            "gt_path": str(gt_path) if gt_path else "",
            "background_path": str(bg_path) if bg_path else "",
            "missing_pred": pred_path is None,
            "missing_gt": gt_path is None,
        }

        if pred_path is None or gt_path is None:
            rows.append(row)
            continue

        try:
            pred_raw, pred_img = binary_arr(pred_path)
            gt, gt_img = binary_arr(gt_path)

            mismatch = geometry_mismatch(pred_img, gt_img)
            any_mismatch = any(mismatch.values())
            row.update({f"geometry_{k}": v for k, v in mismatch.items()})
            row["geometry_any_mismatch"] = any_mismatch

            lcc = keep_largest_component(pred_raw)
            raw_d = dice(pred_raw, gt)
            lcc_d = dice(lcc, gt)
            raw_p, raw_r = precision_recall(pred_raw, gt)
            lcc_p, lcc_r = precision_recall(lcc, gt)

            spacing = spacing_tuple(gt_img)
            gt_cent = centroid_index(gt)
            raw_cent = centroid_index(pred_raw)
            lcc_cent = centroid_index(lcc)
            raw_dist = centroid_mm_distance(gt_cent, raw_cent, spacing)
            lcc_dist = centroid_mm_distance(gt_cent, lcc_cent, spacing)
            raw_x_dist = centroid_x_mm_distance(gt_cent, raw_cent, spacing)
            lcc_x_dist = centroid_x_mm_distance(gt_cent, lcc_cent, spacing)

            raw_opp = opposite_image_half(gt_cent, raw_cent, gt.shape, spacing)
            lcc_opp = opposite_image_half(gt_cent, lcc_cent, gt.shape, spacing)

            raw_stats = component_stats(pred_raw)
            lcc_stats = component_stats(lcc)

            row.update({
                "canonical_v2_raw_dice": raw_d,
                "canonical_v2_lcc_dice": lcc_d,
                "canonical_v2_raw_precision": raw_p,
                "canonical_v2_raw_recall": raw_r,
                "canonical_v2_lcc_precision": lcc_p,
                "canonical_v2_lcc_recall": lcc_r,
                "canonical_lcc_collapse": bool(raw_d > RAW_DICE_COLLAPSE_MIN and lcc_d < LCC_DICE_COLLAPSE_MAX),
                "canonical_lcc_opposite_half": bool(lcc_opp),
                "canonical_raw_opposite_half": bool(raw_opp),
                "canonical_lcc_centroid_distance_mm": lcc_dist,
                "canonical_raw_centroid_distance_mm": raw_dist,
                "canonical_lcc_centroid_x_distance_mm": lcc_x_dist,
                "canonical_raw_centroid_x_distance_mm": raw_x_dist,
                "canonical_lcc_centroid_gt_100mm": bool(np.isfinite(lcc_dist) and lcc_dist > LARGE_CENTROID_DISTANCE_MM),
                "canonical_raw_centroid_gt_100mm": bool(np.isfinite(raw_dist) and raw_dist > LARGE_CENTROID_DISTANCE_MM),
                "canonical_gt_voxels": int(gt.sum()),
                "canonical_raw_voxels": int(pred_raw.sum()),
                "canonical_lcc_voxels": int(lcc.sum()),
            })
            row.update({f"canonical_raw_{k}": v for k, v in raw_stats.items()})
            row.update({f"canonical_lcc_{k}": v for k, v in lcc_stats.items()})

            # Copy minimal NIfTI review packet and preview for union cases after flags known
            rows.append(row)

            if i % 50 == 0:
                print(f"  canonical audit {i}/{len(case_ids)}")
        except Exception as e:
            row["error"] = repr(e)
            rows.append(row)
            continue

    df = pd.DataFrame(rows)

    # Merge selected metadata from base, without overwriting canonical columns.
    base_keep = base.copy()
    # Keep columns likely useful if present.
    useful = [
        "case_id_std", "v2_lcc_volume_bound_failure_signature",
        "volume_bound_diagnostic_group", "v2_lcc_theoretical_max_dice_by_volume_ratio",
        "v2_lcc_dice_gap_to_volume_bound", "v2_lcc_dice_utilization_of_volume_bound",
        "v2_lcc_pred_to_gt_vol_ratio", "gt_vol_cm3", "gt_volume_quintile",
        "overall", "strict_phase_group", "early_t_s", "late_t_s", "early_dev_s", "late_dev_s",
    ]
    useful = [c for c in useful if c in base_keep.columns]
    if useful:
        df = df.merge(base_keep[useful].drop_duplicates("case_id_std"), on="case_id_std", how="left")

    # Pattern groups
    df["canonical_pattern_group"] = np.select(
        [
            df["canonical_lcc_collapse"].fillna(False) & df["canonical_lcc_opposite_half"].fillna(False),
            df["canonical_lcc_collapse"].fillna(False) & ~df["canonical_lcc_opposite_half"].fillna(False),
            ~df["canonical_lcc_collapse"].fillna(False) & df["canonical_lcc_opposite_half"].fillna(False),
        ],
        ["collapse_intersect_opposite", "collapse_not_opposite", "opposite_not_collapse"],
        default="neither",
    )

    # Save all-case canonical result
    all_path = METRICS_DIR / "canonical_all_case_lcc_opposite_audit.csv"
    df.to_csv(all_path, index=False)
    print(f"Saved canonical all-case audit: {all_path}")

    # Counts and exact tests
    collapse = df["canonical_lcc_collapse"].fillna(False)
    opposite = df["canonical_lcc_opposite_half"].fillna(False)
    centroid100 = df["canonical_lcc_centroid_gt_100mm"].fillna(False)

    a = int((collapse & opposite).sum())
    b = int((collapse & ~opposite).sum())
    c = int((~collapse & opposite).sum())
    d = int((~collapse & ~opposite).sum())
    or_val = odds_ratio_from_counts(a, b, c, d)

    fisher_p = np.nan
    chi2_p = np.nan
    if fisher_exact is not None:
        try:
            or_scipy, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            # use scipy OR if finite
            if np.isfinite(or_scipy):
                or_val = float(or_scipy)
        except Exception:
            pass
    if chi2_contingency is not None:
        try:
            chi2, chi2_p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
        except Exception:
            pass

    counts = {
        "all_cases": int(len(df)),
        "lcc_collapse": int(collapse.sum()),
        "opposite_half": int(opposite.sum()),
        "centroid_gt_100mm": int(centroid100.sum()),
        "collapse_intersect_opposite": a,
        "collapse_not_opposite": b,
        "opposite_not_collapse": c,
        "neither": d,
        "collapse_intersect_centroid100": int((collapse & centroid100).sum()),
        "opposite_intersect_centroid100": int((opposite & centroid100).sum()),
        "all_three_collapse_opposite_centroid100": int((collapse & opposite & centroid100).sum()),
        "union_collapse_or_opposite": int((collapse | opposite).sum()),
        "collapse_set_equals_opposite_set": bool(set(df.loc[collapse, "case_id_std"]) == set(df.loc[opposite, "case_id_std"])),
        "odds_ratio_collapse_vs_opposite": or_val,
        "fisher_exact_p": fisher_p,
        "chi2_p": chi2_p,
    }
    pd.DataFrame([counts]).to_csv(METRICS_DIR / "canonical_lcc_opposite_intersection_summary.csv", index=False)

    ctab = pd.DataFrame(
        [[a, b], [c, d]],
        index=["LCC-collapse", "not LCC-collapse"],
        columns=["opposite-half", "not opposite-half"],
    )
    ctab.to_csv(METRICS_DIR / "canonical_crosstab_lcc_collapse_vs_opposite_half.csv")

    # Case lists
    df.loc[collapse & opposite].to_csv(METRICS_DIR / "canonical_case_list_collapse_intersect_opposite.csv", index=False)
    df.loc[collapse & ~opposite].to_csv(METRICS_DIR / "canonical_case_list_collapse_not_opposite.csv", index=False)
    df.loc[~collapse & opposite].to_csv(METRICS_DIR / "canonical_case_list_opposite_not_collapse.csv", index=False)
    df.loc[collapse | opposite].to_csv(METRICS_DIR / "canonical_case_list_union_collapse_or_opposite.csv", index=False)

    # Group and cohort summaries
    group_summary(df, "canonical_pattern_group").to_csv(METRICS_DIR / "canonical_profile_by_pattern_group.csv", index=False)
    cohort_group = df.groupby(["cohort", "canonical_pattern_group"], dropna=False).size().reset_index(name="n")
    cohort_group.to_csv(METRICS_DIR / "canonical_cohort_breakdown_by_pattern_group.csv", index=False)

    # Compare previous D13/D14/D15 flags to canonical
    comparisons = [
        compare_previous_sets(df, D13_TABLE, "D13"),
        compare_previous_sets(df, D14_TABLE, "D14"),
        compare_previous_sets(df, D15_TABLE, "D15"),
    ]
    pd.DataFrame(comparisons).to_csv(METRICS_DIR / "previous_audit_flag_set_differences_vs_canonical.csv", index=False)

    # Review packet from exact canonical union
    union_df = df.loc[collapse | opposite].copy()
    union_ids = sorted(union_df["case_id_std"].tolist())
    preview_meta_rows = []
    preview_paths = []

    print(f"Creating corrected review packet for canonical union cases: {len(union_ids)}")
    for j, case_id in enumerate(union_ids, 1):
        pred_path = find_pred_path(case_id, PRED_V2_DIR)
        gt_path = find_gt_path(case_id, GT_DIR)
        bg_path = find_background_image(case_id)
        if pred_path is None or gt_path is None:
            continue
        try:
            pred_raw, pred_img = binary_arr(pred_path)
            gt, gt_img = binary_arr(gt_path)
            lcc = keep_largest_component(pred_raw)
            bg_arr = None
            if bg_path is not None and bg_path.exists():
                bg_arr = img_to_arr(load_sitk(bg_path))
                # If background shape mismatches, ignore it.
                if bg_arr.shape != gt.shape:
                    bg_arr = None

            row = df.loc[df["case_id_std"] == case_id].iloc[0].to_dict()
            out_png = PREVIEW_DIR / f"{case_id}_canonical_preview.png"
            make_preview(case_id, bg_arr, gt, pred_raw, lcc, row, out_png)
            preview_paths.append(out_png)

            if SAVE_NIFTI_PACKET:
                case_dir = NIFTI_DIR / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                # Save raw/lcc/gt and background if available
                sitk.WriteImage(sitk.GetImageFromArray(gt.astype(np.uint8)), str(case_dir / f"{case_id}_gt_mask_arrayspace.nii.gz"), useCompression=True)
                sitk.WriteImage(sitk.GetImageFromArray(pred_raw.astype(np.uint8)), str(case_dir / f"{case_id}_v2_raw_pred_arrayspace.nii.gz"), useCompression=True)
                sitk.WriteImage(sitk.GetImageFromArray(lcc.astype(np.uint8)), str(case_dir / f"{case_id}_v2_lcc_pred_arrayspace.nii.gz"), useCompression=True)
                # Also copy originals with geometry if possible
                try:
                    import shutil
                    shutil.copy2(str(gt_path), str(case_dir / f"{case_id}_gt_mask_original.nii.gz"))
                    shutil.copy2(str(pred_path), str(case_dir / f"{case_id}_v2_raw_pred_original.nii.gz"))
                    if bg_path is not None and bg_path.exists():
                        shutil.copy2(str(bg_path), str(case_dir / f"{case_id}_background_pre_or_input0000.nii.gz"))
                except Exception:
                    pass

            preview_meta_rows.append({
                "case_id_std": case_id,
                "preview_path": str(out_png),
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "background_path": str(bg_path) if bg_path else "",
                "canonical_lcc_collapse": bool(row.get("canonical_lcc_collapse", False)),
                "canonical_lcc_opposite_half": bool(row.get("canonical_lcc_opposite_half", False)),
                "canonical_pattern_group": row.get("canonical_pattern_group", ""),
                "raw_dice": row.get("canonical_v2_raw_dice", np.nan),
                "lcc_dice": row.get("canonical_v2_lcc_dice", np.nan),
                "lcc_centroid_distance_mm": row.get("canonical_lcc_centroid_distance_mm", np.nan),
            })
        except Exception as e:
            preview_meta_rows.append({
                "case_id_std": case_id,
                "preview_error": repr(e),
            })
        if j % 25 == 0:
            print(f"  previews {j}/{len(union_ids)}")

    pd.DataFrame(preview_meta_rows).to_csv(METRICS_DIR / "canonical_review_packet_manifest.csv", index=False)
    make_contact_sheet(
        preview_paths,
        FIGURES_DIR / "contact_sheet_canonical_union_collapse_or_opposite.jpg",
        "Canonical review packet: LCC-collapse OR opposite-image-half",
    )

    # Figures: simple bar counts
    try:
        count_df = df["canonical_pattern_group"].value_counts().reset_index()
        count_df.columns = ["pattern_group", "n"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(count_df["pattern_group"], count_df["n"])
        ax.set_xlabel("Number of cases")
        ax.set_title("Canonical LCC-collapse / opposite-half pattern groups")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig01_canonical_pattern_group_counts.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Write takeaways
    takeaways = []
    takeaways.append("15b CANONICAL LCC-COLLAPSE / OPPOSITE-HALF AUDIT")
    takeaways.append("=" * 72)
    takeaways.append(f"Cases audited: {len(df)}")
    takeaways.append("")
    takeaways.append("Canonical flag definitions:")
    takeaways.append(f"  LCC-collapse: raw Dice > {RAW_DICE_COLLAPSE_MIN} and LCC Dice < {LCC_DICE_COLLAPSE_MAX}")
    takeaways.append(f"  Opposite-image-half: GT and LCC centroids on opposite image halves with x-distance >= {OPPOSITE_MIN_X_DISTANCE_MM} mm")
    takeaways.append("")
    takeaways.append("Canonical counts:")
    for k in [
        "lcc_collapse", "opposite_half", "centroid_gt_100mm",
        "collapse_intersect_opposite", "collapse_not_opposite",
        "opposite_not_collapse", "union_collapse_or_opposite",
        "collapse_set_equals_opposite_set",
    ]:
        takeaways.append(f"  {k}: {counts[k]}")
    takeaways.append("")
    takeaways.append(f"Odds ratio collapse vs opposite-half: {counts['odds_ratio_collapse_vs_opposite']:.4g}")
    takeaways.append(f"Fisher exact p: {counts['fisher_exact_p']:.4g}")
    takeaways.append(f"Chi-square p: {counts['chi2_p']:.4g}")
    takeaways.append("")
    takeaways.append("Interpretation:")
    takeaways.append("  Opposite-image-half / large-displacement component selection is associated with LCC-collapse,")
    takeaways.append("  but it is not identical to LCC-collapse and does not explain all collapse cases.")
    takeaways.append("  Use cautious wording: 'associated with' or 'observed in a subset', not 'caused by'.")
    takeaways.append("")
    takeaways.append("Review-packet note:")
    takeaways.append(f"  Corrected review packet generated from exact canonical union, n={len(union_ids)}.")
    takeaways.append("  Use canonical_review_packet_manifest.csv and the contact sheet for visual review.")
    takeaways.append("")
    takeaways.append("Previous audit comparison:")
    for comp in comparisons:
        takeaways.append(f"  {comp.get('source')}: status={comp.get('status')}, collapse_col={comp.get('collapse_col_used')}, opposite_col={comp.get('opposite_col_used')}")
        for k in ["prev_collapse_n", "canonical_collapse_n", "collapse_intersection", "collapse_prev_minus_canonical", "collapse_canonical_minus_prev", "prev_opposite_n", "canonical_opposite_n", "opposite_intersection", "opposite_prev_minus_canonical", "opposite_canonical_minus_prev"]:
            if k in comp:
                takeaways.append(f"    {k}: {comp[k]}")
    (OUTPUT_ROOT / "article_takeaways_15b.txt").write_text("\n".join(takeaways), encoding="utf-8")

    print()
    print("DONE — 15b canonical audit complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Previews: {PREVIEW_DIR}")
    print(f"Takeaways: {OUTPUT_ROOT / 'article_takeaways_15b.txt'}")
    print()
    print("Canonical quick results:")
    print(f"  all cases: {len(df)}")
    print(f"  LCC-collapse: {counts['lcc_collapse']}")
    print(f"  opposite-half: {counts['opposite_half']}")
    print(f"  collapse ∩ opposite: {counts['collapse_intersect_opposite']}")
    print(f"  collapse outside opposite: {counts['collapse_not_opposite']}")
    print(f"  opposite outside collapse: {counts['opposite_not_collapse']}")
    print(f"  union(collapse, opposite): {counts['union_collapse_or_opposite']}")
    print(f"  collapse set == opposite set: {counts['collapse_set_equals_opposite_set']}")
    print(f"  OR: {counts['odds_ratio_collapse_vs_opposite']:.4g}")
    print(f"  Fisher p: {counts['fisher_exact_p']:.4g}")
    print()
    print("Open:")
    print(f"  {FIGURES_DIR / 'contact_sheet_canonical_union_collapse_or_opposite.jpg'}")
    print(f"  {METRICS_DIR / 'canonical_review_packet_manifest.csv'}")


if __name__ == "__main__":
    main()
