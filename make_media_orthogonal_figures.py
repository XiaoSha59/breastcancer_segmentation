#!/usr/bin/env python
"""
Create publication-ready orthogonal DCE-MRI segmentation figures for MedIA.

The script uses the local folder structure already referenced by MRI-Project:

    PROJECT_ROOT/
      images_2/<CASE_ID>/*_0000.nii.gz ... *_0004.nii.gz
      segmentations_2/expert/<case_id_lower>.nii.gz
      10_final_external_sanity_checks/metrics/
          final_sanity_case_table_with_volume_bounds.csv

and the raw timing-aware predictions:

    C:/nnw/mama_mia_output_v2/<CASE_ID>.nii.gz

Outputs:
    12_media_orthogonal_figures/
      media_typical_case_orthogonal.png
      media_typical_case_orthogonal.tiff
      media_lcc_collapse_case_orthogonal.png
      media_lcc_collapse_case_orthogonal.tiff
      media_combined_full_fov_orthogonal.png
      selected_cases_for_media_figure.csv

The separate figures contain:
    row 1: full-field-of-view axial, coronal, and sagittal views
    row 2: zoomed versions of the same three planes

The combined figure contains full-field-of-view views for both a typical case
and an LCC-collapse case.

Contours:
    Expert mask = green
    Raw prediction = red
    LCC prediction = yellow

Requirements:
    pip install numpy pandas scipy matplotlib SimpleITK
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------------------------------------------------------------
# DEFAULT PATHS — match the existing MRI-Project scripts
# -------------------------------------------------------------------------
DEFAULT_PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")
DEFAULT_PRED_DIR = Path(r"C:\nnw\mama_mia_output_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate orthogonal full-FOV qualitative figures for MedIA."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Local MRI Project folder.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Folder containing timing-aware raw predictions as <CASE_ID>.nii.gz.",
    )
    parser.add_argument(
        "--typical-case",
        type=str,
        default=None,
        help="Optional explicit representative case ID, e.g. DUKE_517.",
    )
    parser.add_argument(
        "--collapse-case",
        type=str,
        default=None,
        help="Optional explicit LCC-collapse case ID.",
    )
    parser.add_argument(
        "--background-channel",
        choices=["pre", "early", "late", "early_sub", "late_sub"],
        default="early",
        help="DCE channel used as grayscale background. Default: early post-contrast.",
    )
    parser.add_argument(
        "--no-auto-select",
        action="store_true",
        help="Disable automatic case selection; explicit case IDs then become mandatory.",
    )
    parser.add_argument(
        "--show-case-ids",
        action="store_true",
        help="Show de-identified dataset case IDs in figure titles. Default: hidden.",
    )
    return parser.parse_args()


CHANNEL_SUFFIX = {
    "pre": "0000",
    "early": "0001",
    "late": "0002",
    "early_sub": "0003",
    "late_sub": "0004",
}


def clean_case_id(value: object) -> str:
    return str(value).strip().replace(".nii.gz", "").replace(".nii", "").upper()


def first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def case_id_column(df: pd.DataFrame) -> str:
    col = first_existing_column(df, ["case_id", "case_id_std", "patient_id", "id"])
    if col is None:
        raise KeyError("No case identifier column was found in the case table.")
    return col


def numeric_column(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    col = first_existing_column(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = df[col]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def load_case_table(project_root: Path) -> pd.DataFrame:
    path = (
        project_root
        / "10_final_external_sanity_checks"
        / "metrics"
        / "final_sanity_case_table_with_volume_bounds.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Case table not found:\n  {path}\n"
            "Run the repository's script 10 first or pass explicit case IDs."
        )

    df = pd.read_csv(path, low_memory=False)
    cid_col = case_id_column(df)
    df["case_id_clean"] = df[cid_col].map(clean_case_id)
    df["raw_dice_for_selection"] = numeric_column(
        df, ["v2_raw_dice", "raw_dice", "dice_raw"]
    )
    df["lcc_dice_for_selection"] = numeric_column(
        df, ["v2_lcc_dice", "lcc_dice", "dice"]
    )
    return df


def select_typical_case(df: pd.DataFrame) -> str:
    """
    Select a non-extreme representative case rather than the best case.

    We choose the case closest to the 75th percentile of finite LCC Dice among
    cases with raw and LCC predictions. This gives a clear visualization while
    avoiding selection of the single highest-scoring case.
    """
    valid = df[
        df["lcc_dice_for_selection"].notna()
        & df["raw_dice_for_selection"].notna()
        & (df["lcc_dice_for_selection"] > 0)
    ].copy()
    if valid.empty:
        raise RuntimeError("No finite non-zero LCC Dice values were found.")

    target = float(valid["lcc_dice_for_selection"].quantile(0.75))
    valid["distance_to_target"] = (
        valid["lcc_dice_for_selection"] - target
    ).abs()
    row = valid.sort_values(
        ["distance_to_target", "lcc_dice_for_selection"],
        ascending=[True, False],
    ).iloc[0]
    return clean_case_id(row["case_id_clean"])


def select_lcc_collapse_case(df: pd.DataFrame) -> str:
    """
    Apply the manuscript's prespecified LCC-collapse definition:
        raw Dice > 0.20 and LCC Dice < 0.05

    Among collapse cases, choose the case with the highest raw Dice so the
    postprocessing-related degradation is clearly visible.
    """
    collapse = df[
        (df["raw_dice_for_selection"] > 0.20)
        & (df["lcc_dice_for_selection"] < 0.05)
    ].copy()
    if collapse.empty:
        raise RuntimeError(
            "No case met raw Dice > 0.20 and LCC Dice < 0.05."
        )
    row = collapse.sort_values(
        ["raw_dice_for_selection", "case_id_clean"],
        ascending=[False, True],
    ).iloc[0]
    return clean_case_id(row["case_id_clean"])


def find_case_dir(images_dir: Path, case_id: str) -> Path:
    direct = images_dir / case_id
    if direct.is_dir():
        return direct
    matches = [
        p for p in images_dir.iterdir()
        if p.is_dir() and p.name.upper() == case_id.upper()
    ] if images_dir.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image folder not found for {case_id}: {direct}")


def extract_channel_number(path: Path) -> Optional[str]:
    match = re.search(r"_(\d{4})\.nii(?:\.gz)?$", path.name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def find_channel_path(case_dir: Path, channel_name: str) -> Path:
    suffix = CHANNEL_SUFFIX[channel_name]
    candidates = sorted(case_dir.glob("*.nii*"))
    for path in candidates:
        if extract_channel_number(path) == suffix:
            return path

    # Safe fallback for datasets that do not retain nnU-Net suffixes.
    if channel_name == "early" and len(candidates) >= 2:
        return candidates[1]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No NIfTI image found in {case_dir}")


def case_paths(
    project_root: Path,
    prediction_dir: Path,
    case_id: str,
    background_channel: str,
) -> Dict[str, Path]:
    case_id = clean_case_id(case_id)
    images_dir = project_root / "images_2"
    gt_dir = project_root / "segmentations_2" / "expert"

    case_dir = find_case_dir(images_dir, case_id)
    background = find_channel_path(case_dir, background_channel)
    gt = gt_dir / f"{case_id.lower()}.nii.gz"
    pred = prediction_dir / f"{case_id}.nii.gz"

    missing = [p for p in [background, gt, pred] if not p.exists()]
    if missing:
        formatted = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(
            f"Required files are missing for {case_id}:\n{formatted}"
        )

    return {
        "background": background,
        "gt": gt,
        "pred_raw": pred,
        "case_dir": case_dir,
    }


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4, rtol=0)
    )


def resample_label_to_reference(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if same_geometry(label, reference):
        return label
    return sitk.Resample(
        label,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def orient_lps(image: sitk.Image) -> sitk.Image:
    """
    Reorient to a consistent LPS anatomical orientation before extracting
    axial, coronal, and sagittal planes.
    """
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image)


def load_case_arrays(paths: Dict[str, Path]) -> Dict[str, object]:
    bg_img = sitk.ReadImage(str(paths["background"]), sitk.sitkFloat32)
    gt_img = sitk.ReadImage(str(paths["gt"]))
    pred_img = sitk.ReadImage(str(paths["pred_raw"]))

    gt_img = resample_label_to_reference(gt_img, bg_img)
    pred_img = resample_label_to_reference(pred_img, bg_img)

    # Reorient every volume identically into LPS.
    bg_img = orient_lps(bg_img)
    gt_img = orient_lps(gt_img)
    pred_img = orient_lps(pred_img)

    bg = sitk.GetArrayFromImage(bg_img).astype(np.float32)
    gt = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
    pred_raw = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
    pred_lcc = keep_largest_component(pred_raw)

    if not (bg.shape == gt.shape == pred_raw.shape == pred_lcc.shape):
        raise RuntimeError(
            "Arrays do not share the same geometry after resampling and orientation: "
            f"background={bg.shape}, gt={gt.shape}, pred={pred_raw.shape}"
        )

    return {
        "background": bg,
        "gt": gt,
        "pred_raw": pred_raw,
        "pred_lcc": pred_lcc,
        "spacing_xyz": tuple(float(v) for v in bg_img.GetSpacing()),
        "origin_xyz": tuple(float(v) for v in bg_img.GetOrigin()),
        "direction": tuple(float(v) for v in bg_img.GetDirection()),
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    labels, count = ndimage.label(mask > 0)
    if count == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask > 0, labels, index=np.arange(1, count + 1))
    largest = int(np.argmax(sizes) + 1)
    return (labels == largest).astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    denominator = int(p.sum() + g.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / denominator)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[finite]
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def best_index(mask_union: np.ndarray, axis: int) -> int:
    reduce_axes = tuple(i for i in range(3) if i != axis)
    areas = mask_union.sum(axis=reduce_axes)
    if np.max(areas) <= 0:
        return int(mask_union.shape[axis] // 2)
    return int(np.argmax(areas))


def plane_at(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:  # axial: z fixed -> y,x
        return volume[index, :, :]
    if axis == 1:  # coronal: y fixed -> z,x
        return volume[:, index, :]
    if axis == 2:  # sagittal: x fixed -> z,y
        return volume[:, :, index]
    raise ValueError(f"Invalid axis: {axis}")


def physical_aspect(spacing_xyz: Tuple[float, float, float], axis: int) -> float:
    sx, sy, sz = spacing_xyz
    if axis == 0:  # y / x
        return sy / sx
    if axis == 1:  # z / x
        return sz / sx
    if axis == 2:  # z / y
        return sz / sy
    return 1.0


def crop_slices(
    masks_2d: Sequence[np.ndarray],
    margin_fraction: float = 0.15,
    minimum_margin: int = 20,
) -> Tuple[slice, slice]:
    coords = []
    for mask in masks_2d:
        where = np.argwhere(mask > 0)
        if where.size:
            coords.append(where)

    if not coords:
        shape = masks_2d[0].shape
        return slice(0, shape[0]), slice(0, shape[1])

    all_coords = np.vstack(coords)
    y0, x0 = all_coords.min(axis=0)
    y1, x1 = all_coords.max(axis=0)

    height = max(int(y1 - y0 + 1), 1)
    width = max(int(x1 - x0 + 1), 1)
    margin_y = max(int(height * margin_fraction), minimum_margin)
    margin_x = max(int(width * margin_fraction), minimum_margin)

    shape_y, shape_x = masks_2d[0].shape
    return (
        slice(max(int(y0) - margin_y, 0), min(int(y1) + margin_y + 1, shape_y)),
        slice(max(int(x0) - margin_x, 0), min(int(x1) + margin_x + 1, shape_x)),
    )


def draw_overlay(
    ax: plt.Axes,
    background: np.ndarray,
    gt: np.ndarray,
    raw: np.ndarray,
    lcc: np.ndarray,
    title: str,
    aspect: float,
    crop: Optional[Tuple[slice, slice]] = None,
) -> None:
    if crop is not None:
        ys, xs = crop
        background = background[ys, xs]
        gt = gt[ys, xs]
        raw = raw[ys, xs]
        lcc = lcc[ys, xs]

    ax.imshow(background, cmap="gray", origin="lower", aspect=aspect)

    if gt.max() > 0:
        ax.contour(gt.astype(float), [0.5], colors=["lime"], linewidths=1.5)
    if raw.max() > 0:
        ax.contour(raw.astype(float), [0.5], colors=["red"], linewidths=1.1)
    if lcc.max() > 0:
        ax.contour(lcc.astype(float), [0.5], colors=["yellow"], linewidths=1.2)

    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")


def case_planes(case: Dict[str, object]) -> Dict[str, object]:
    background = normalize_volume(case["background"])
    gt = case["gt"]
    raw = case["pred_raw"]
    lcc = case["pred_lcc"]
    union = ((gt > 0) | (raw > 0) | (lcc > 0)).astype(np.uint8)

    plane_names = ["Axial", "Coronal", "Sagittal"]
    axes = [0, 1, 2]
    indices = [best_index(union, axis) for axis in axes]

    result = {}
    for name, axis, index in zip(plane_names, axes, indices):
        bg2d = plane_at(background, axis, index)
        gt2d = plane_at(gt, axis, index)
        raw2d = plane_at(raw, axis, index)
        lcc2d = plane_at(lcc, axis, index)
        crop = crop_slices([gt2d, raw2d, lcc2d])
        result[name] = {
            "axis": axis,
            "index": index,
            "background": bg2d,
            "gt": gt2d,
            "raw": raw2d,
            "lcc": lcc2d,
            "crop": crop,
            "aspect": physical_aspect(case["spacing_xyz"], axis),
        }
    return result


def figure_orthogonal_case(
    case_id: str,
    case: Dict[str, object],
    output_stem: Path,
    figure_label: str,
    show_case_ids: bool = False,
) -> Dict[str, float]:
    planes = case_planes(case)
    raw_dice = dice(case["pred_raw"], case["gt"])
    lcc_dice = dice(case["pred_lcc"], case["gt"])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.3))
    ordered = ["Axial", "Coronal", "Sagittal"]

    for col, name in enumerate(ordered):
        data = planes[name]
        draw_overlay(
            axes[0, col],
            data["background"],
            data["gt"],
            data["raw"],
            data["lcc"],
            f"{name}, full field of view",
            data["aspect"],
            crop=None,
        )
        draw_overlay(
            axes[1, col],
            data["background"],
            data["gt"],
            data["raw"],
            data["lcc"],
            f"{name}, magnified view",
            data["aspect"],
            crop=data["crop"],
        )

    legend = [
        Line2D([0], [0], color="lime", lw=2, label="Expert annotation"),
        Line2D([0], [0], color="red", lw=2, label="Raw prediction"),
        Line2D([0], [0], color="yellow", lw=2, label="Largest connected component"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    case_text = f" | case {case_id}" if show_case_ids else ""
    fig.suptitle(
        f"{figure_label}: volumetric segmentation across orthogonal planes{case_text}\n"
        f"raw Dice={raw_dice:.3f} | LCC Dice={lcc_dice:.3f}",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    return {"raw_dice": raw_dice, "lcc_dice": lcc_dice}


def draw_full_fov_row(
    axes: Sequence[plt.Axes],
    case: Dict[str, object],
    row_label: str,
) -> None:
    planes = case_planes(case)
    ordered = ["Axial", "Coronal", "Sagittal"]
    for col, name in enumerate(ordered):
        data = planes[name]
        draw_overlay(
            axes[col],
            data["background"],
            data["gt"],
            data["raw"],
            data["lcc"],
            f"{name}, full field of view",
            data["aspect"],
        )
    axes[0].text(
        -0.08,
        0.5,
        row_label,
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )


def combined_figure(
    typical_id: str,
    typical: Dict[str, object],
    collapse_id: str,
    collapse: Dict[str, object],
    output_path: Path,
    show_case_ids: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    t_raw = dice(typical["pred_raw"], typical["gt"])
    t_lcc = dice(typical["pred_lcc"], typical["gt"])
    c_raw = dice(collapse["pred_raw"], collapse["gt"])
    c_lcc = dice(collapse["pred_lcc"], collapse["gt"])

    typical_case_text = f"\n{typical_id}" if show_case_ids else ""
    collapse_case_text = f"\n{collapse_id}" if show_case_ids else ""
    draw_full_fov_row(
        axes[0],
        typical,
        f"Representative case{typical_case_text}\nraw={t_raw:.3f}, LCC={t_lcc:.3f}",
    )
    draw_full_fov_row(
        axes[1],
        collapse,
        f"LCC-collapse case{collapse_case_text}\nraw={c_raw:.3f}, LCC={c_lcc:.3f}",
    )

    legend = [
        Line2D([0], [0], color="lime", lw=2, label="Expert annotation"),
        Line2D([0], [0], color="red", lw=2, label="Raw prediction"),
        Line2D([0], [0], color="yellow", lw=2, label="Largest connected component"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Full-field-of-view volumetric segmentation findings in three orthogonal planes",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.035, 0.06, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        output_path.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    prediction_dir = args.prediction_dir
    output_dir = project_root / "12_media_orthogonal_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_auto_select and (not args.typical_case or not args.collapse_case):
        raise ValueError(
            "--no-auto-select requires --typical-case and --collapse-case."
        )

    df: Optional[pd.DataFrame] = None
    if not args.no_auto_select or not args.typical_case or not args.collapse_case:
        df = load_case_table(project_root)

    typical_id = (
        clean_case_id(args.typical_case)
        if args.typical_case
        else select_typical_case(df)
    )
    collapse_id = (
        clean_case_id(args.collapse_case)
        if args.collapse_case
        else select_lcc_collapse_case(df)
    )

    print("Selected cases")
    print("==============")
    print(f"Representative case: {typical_id}")
    print(f"LCC-collapse case  : {collapse_id}")
    print()

    typical_paths = case_paths(
        project_root, prediction_dir, typical_id, args.background_channel
    )
    collapse_paths = case_paths(
        project_root, prediction_dir, collapse_id, args.background_channel
    )

    typical = load_case_arrays(typical_paths)
    collapse = load_case_arrays(collapse_paths)

    typical_metrics = figure_orthogonal_case(
        typical_id,
        typical,
        output_dir / "media_typical_case_orthogonal",
        "Representative external case",
        show_case_ids=args.show_case_ids,
    )
    collapse_metrics = figure_orthogonal_case(
        collapse_id,
        collapse,
        output_dir / "media_lcc_collapse_case_orthogonal",
        "Postprocessing-collapse example",
        show_case_ids=args.show_case_ids,
    )
    combined_figure(
        typical_id,
        typical,
        collapse_id,
        collapse,
        output_dir / "media_combined_full_fov_orthogonal.png",
        show_case_ids=args.show_case_ids,
    )

    selection = pd.DataFrame(
        [
            {
                "role": "representative",
                "case_id": typical_id,
                "raw_dice_computed": typical_metrics["raw_dice"],
                "lcc_dice_computed": typical_metrics["lcc_dice"],
                "selection_rule": "closest to 75th percentile of finite non-zero LCC Dice",
                "background_channel": args.background_channel,
            },
            {
                "role": "lcc_collapse",
                "case_id": collapse_id,
                "raw_dice_computed": collapse_metrics["raw_dice"],
                "lcc_dice_computed": collapse_metrics["lcc_dice"],
                "selection_rule": "highest raw Dice among raw Dice > 0.20 and LCC Dice < 0.05",
                "background_channel": args.background_channel,
            },
        ]
    )
    selection.to_csv(
        output_dir / "selected_cases_for_media_figure.csv", index=False
    )

    print("Saved")
    print("=====")
    for path in sorted(output_dir.glob("media_*")):
        print(path)
    print(output_dir / "selected_cases_for_media_figure.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR")
        print("=====")
        print(exc)
        print(
            "\nCheck PROJECT_ROOT, the prediction directory, and whether the local "
            "NIfTI folders excluded from GitHub are present."
        )
        sys.exit(1)
