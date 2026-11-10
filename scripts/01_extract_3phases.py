"""Data curation & 5-channel preparation for BreastDCEDL I-SPY2 on Linux GCP VM.

Extracts I_0, I_e, I_l, Sub_early, Sub_late + Mask for all 982 patients.
Generates nnU-Net v2 dataset.json, splits_final.json, and metadata_curated.csv.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

BASE_DIR = Path("/home/XiaoSha")
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DATASET_DIR = BASE_DIR / "data" / "processed" / "Dataset501_BreastDCE_ISPY2"
IMAGES_TR_DIR = OUT_DATASET_DIR / "imagesTr"
LABELS_TR_DIR = OUT_DATASET_DIR / "labelsTr"

CASES_CSV = BASE_DIR / "project" / "references" / "metadata" / "cases_all.csv"
NAOMI_CSV = BASE_DIR / "project" / "references" / "metadata" / "BreastDCEDL_ISPY2_metadata.csv"


def find_file(pid: str, subfolder: str, pattern: str) -> Path | None:
    pdir = RAW_DIR / pid / subfolder
    if not pdir.is_dir():
        # Fallback search anywhere under RAW_DIR / pid
        fallback_dir = RAW_DIR / pid
        matches = list(fallback_dir.glob(f"**/{pattern}"))
        return matches[0] if matches else None

    matches = list(pdir.glob(pattern))
    return matches[0] if matches else None


def process_patient(row: dict) -> dict:
    pid = row["patient_id"].strip()
    pre_tp = row["pre_tp"].strip()
    early_tp = row["early_tp"].strip()
    late_tp = row["late_tp"].strip()
    fold = int(row["fold"].strip())

    # Destination paths
    out_img_0000 = IMAGES_TR_DIR / f"{pid}_0000.nii.gz"
    out_img_0001 = IMAGES_TR_DIR / f"{pid}_0001.nii.gz"
    out_img_0002 = IMAGES_TR_DIR / f"{pid}_0002.nii.gz"
    out_img_0003 = IMAGES_TR_DIR / f"{pid}_0003.nii.gz"
    out_img_0004 = IMAGES_TR_DIR / f"{pid}_0004.nii.gz"
    out_label = LABELS_TR_DIR / f"{pid}.nii.gz"

    # Skip if already processed
    if (
        out_img_0000.exists()
        and out_img_0001.exists()
        and out_img_0002.exists()
        and out_img_0003.exists()
        and out_img_0004.exists()
        and out_label.exists()
    ):
        return {"pid": pid, "status": "skipped", "fold": fold}

    pre_file = find_file(pid, "dce", f"*{pid}*dce_aqc_{pre_tp}.nii.gz")
    early_file = find_file(pid, "dce", f"*{pid}*dce_aqc_{early_tp}.nii.gz")
    late_file = find_file(pid, "dce", f"*{pid}*dce_aqc_{late_tp}.nii.gz")
    mask_file = find_file(pid, "mask", f"*{pid}*mask.nii.gz")

    if not (pre_file and early_file and late_file and mask_file):
        return {
            "pid": pid,
            "status": "error",
            "message": f"Missing files: pre={pre_file}, early={early_file}, late={late_file}, mask={mask_file}",
            "fold": fold,
        }

    try:
        # Load NIfTI files
        pre_nii = nib.load(pre_file)
        early_nii = nib.load(early_file)
        late_nii = nib.load(late_file)
        mask_nii = nib.load(mask_file)

        pre_arr = pre_nii.get_fdata(dtype=np.float32)
        early_arr = early_nii.get_fdata(dtype=np.float32)
        late_arr = late_nii.get_fdata(dtype=np.float32)
        mask_arr = (mask_nii.get_fdata() > 0).astype(np.uint8)

        affine = pre_nii.affine
        header = pre_nii.header.copy()

        # Compute Subtraction images
        sub_early = early_arr - pre_arr
        sub_late = late_arr - pre_arr

        # Save 5 channels
        nib.save(nib.Nifti1Image(pre_arr, affine, header), out_img_0000)
        nib.save(nib.Nifti1Image(early_arr, affine, header), out_img_0001)
        nib.save(nib.Nifti1Image(late_arr, affine, header), out_img_0002)
        nib.save(nib.Nifti1Image(sub_early, affine, header), out_img_0003)
        nib.save(nib.Nifti1Image(sub_late, affine, header), out_img_0004)

        # Save Label
        nib.save(nib.Nifti1Image(mask_arr, affine), out_label)

        shape_str = str(pre_arr.shape)
        voxel_vol = int(np.sum(mask_arr))

        return {
            "pid": pid,
            "status": "success",
            "fold": fold,
            "shape": shape_str,
            "tumor_voxels": voxel_vol,
        }
    except Exception as e:
        return {"pid": pid, "status": "error", "message": str(e), "fold": fold}


def main() -> None:
    print("=" * 65, flush=True)
    print("=== STARTING 5-CHANNEL DATA EXTRACTION & CURATION ===", flush=True)
    print("=" * 65, flush=True)

    IMAGES_TR_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_TR_DIR.mkdir(parents=True, exist_ok=True)

    if not CASES_CSV.exists():
        print(f"Error: {CASES_CSV} not found!", flush=True)
        sys.exit(1)

    with open(CASES_CSV, "r", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    print(f"Total cases in manifest: {len(cases)}", flush=True)

    start_time = time.time()
    results = []

    # Execute in parallel using 6 workers
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_patient, row): row for row in cases}
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 50 == 0 or done_count == len(cases):
                elapsed = time.time() - start_time
                print(f"  Processed [{done_count}/{len(cases)}] patients ({elapsed:.1f}s)", flush=True)

    success_cases = [r for r in results if r["status"] in ("success", "skipped")]
    error_cases = [r for r in results if r["status"] == "error"]

    print("\n" + "=" * 65, flush=True)
    print(f"Processing Summary:", flush=True)
    print(f"  Success: {len(success_cases)} / {len(cases)}", flush=True)
    print(f"  Errors : {len(error_cases)}", flush=True)
    if error_cases:
        for err in error_cases[:5]:
            print(f"    Sample error: {err}", flush=True)
    print("=" * 65, flush=True)

    # ── GENERATE SPLITS_FINAL.JSON (5-Fold CV) ─────────────────────────────────
    print("\nGenerating nnU-Net splits_final.json...", flush=True)
    folds_dict: dict[int, list[str]] = {i: [] for i in range(5)}
    for r in success_cases:
        folds_dict[r["fold"]].append(r["pid"])

    splits = []
    for val_fold in range(5):
        train_cases = []
        for f_idx in range(5):
            if f_idx != val_fold:
                train_cases.extend(folds_dict[f_idx])
        val_cases = folds_dict[val_fold]
        splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})

    splits_file = OUT_DATASET_DIR / "splits_final.json"
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved: {splits_file}", flush=True)
    for i, s in enumerate(splits):
        print(f"  Fold {i}: Train={len(s['train'])}, Val={len(s['val'])}", flush=True)

    # ── GENERATE DATASET.JSON ──────────────────────────────────────────────────
    print("\nGenerating dataset.json (nnU-Net v2 format)...", flush=True)
    dataset_json = {
        "channel_names": {
            "0": "Pre",
            "1": "Early",
            "2": "Late",
            "3": "SubEarly",
            "4": "SubLate",
        },
        "labels": {
            "background": 0,
            "tumor": 1,
        },
        "numTraining": len(success_cases),
        "file_ending": ".nii.gz",
        "name": "Dataset501_BreastDCE_ISPY2",
        "description": "I-SPY2 3D Breast DCE-MRI 5-Channel Tumor Segmentation & HER2 Classification Benchmark",
        "reference": "Elena Krikun et al. (EJR AI 2026) / Naomi Fridman et al. (Nature Sci Data 2026)",
    }
    dataset_json_file = OUT_DATASET_DIR / "dataset.json"
    with open(dataset_json_file, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Saved: {dataset_json_file}", flush=True)

    # ── MERGE WITH NAOMI'S METADATA FOR MULTITASK ──────────────────────────────
    if NAOMI_CSV.exists():
        print("\nMerging with Naomi's clinical metadata (HER2, HR, pCR)...", flush=True)
        naomi_df = pd.read_csv(NAOMI_CSV)
        naomi_df["pid"] = naomi_df["pid"].str.strip()

        curated_rows = []
        for r in success_cases:
            pid = r["pid"]
            fold = r["fold"]
            matched = naomi_df[naomi_df["pid"] == pid]
            row_dict = {
                "patient_id": pid,
                "fold": fold,
                "shape": r.get("shape", ""),
                "tumor_voxels": r.get("tumor_voxels", 0),
            }
            if not matched.empty:
                m = matched.iloc[0]
                row_dict["HER2"] = m.get("HER2", np.nan)
                row_dict["HR"] = m.get("HR", np.nan)
                row_dict["pCR"] = m.get("pCR", np.nan)
                row_dict["tum_vol_mm3"] = m.get("tum_vol", np.nan)
                row_dict["age"] = m.get("age", np.nan)
                row_dict["subtype"] = m.get("HR_HER2_STATUS", "")
            curated_rows.append(row_dict)

        curated_df = pd.DataFrame(curated_rows)
        curated_csv_file = OUT_DATASET_DIR / "metadata_curated.csv"
        curated_df.to_csv(curated_csv_file, index=False)
        print(f"Saved: {curated_csv_file} ({len(curated_df)} rows)", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 65, flush=True)
    print(f"DATA EXTRACTION COMPLETE in {total_time/60:.2f} minutes!", flush=True)
    print(f"Destination: {OUT_DATASET_DIR}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
