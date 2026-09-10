#!/usr/bin/env python3
"""Reconstruct the leakage-controlled external MAMA-MIA cohort.

This reviewer-facing script reproduces the overlap-removal step described in
the manuscript. It compares normalized patient identifiers between the
BreastDCEDL/I-SPY2 development roster and the full MAMA-MIA clinical/imaging
table, excludes matched I-SPY2-source cases, and verifies the committed
526-case external manifest.

No MRI images are required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


EXPECTED = {
    "mama_total": 1506,
    "development_total": 982,
    "overlap": 980,
    "external_total": 526,
    "cohorts": {"DUKE": 291, "ISPY1": 171, "NACT": 64},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_mama_id(value: object) -> str:
    """Normalize a MAMA-MIA patient identifier without discarding cohort prefix."""
    s = str(value).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "_")
    s = re.sub(r"_+", "_", s)

    # Common MAMA-MIA I-SPY2 spellings -> ISPY2_<numeric patient key>
    m = re.search(r"(?:ISPY2|I_SPY2)[_]?(\d+)$", s)
    if m:
        return f"ISPY2_{m.group(1)}"

    return s


def normalize_development_id(value: object) -> str:
    """Map BreastDCEDL/I-SPY2 identifiers to the MAMA-MIA ISPY2_<id> form."""
    s = str(value).strip().upper()
    s = re.sub(r"\s+", "", s)

    # BreastDCEDL uses forms such as ACRIN-6698-102212.
    m = re.search(r"(\d+)$", s)
    if not m:
        raise ValueError(f"Could not extract numeric I-SPY2 patient key from: {value!r}")
    return f"ISPY2_{m.group(1)}"


def load_mama_table(path: Path) -> tuple[pd.DataFrame, str]:
    """Find the workbook sheet containing patient_id and dataset columns."""
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        header = pd.read_excel(xls, sheet_name=sheet, nrows=0)
        cols = {str(c).strip().lower() for c in header.columns}
        if {"patient_id", "dataset"}.issubset(cols):
            df = pd.read_excel(xls, sheet_name=sheet)
            return df, sheet
    raise RuntimeError(
        f"No sheet in {path} contains both 'patient_id' and 'dataset'. "
        f"Sheets: {xls.sheet_names}"
    )


def require(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}" + (f" -- {detail}" if detail else ""))
        raise AssertionError(label)


def compare_id_sets(
    generated: pd.DataFrame,
    committed_path: Path,
    label: str,
) -> None:
    if not committed_path.exists():
        print(f"[WARN] {label}: committed file not found: {committed_path}")
        return

    committed = pd.read_csv(committed_path)
    require(
        "patient_id" in committed.columns,
        f"{label}: committed file has patient_id column",
    )

    generated_ids = set(generated["patient_id"].astype(str).str.strip())
    committed_ids = set(committed["patient_id"].astype(str).str.strip())

    missing = sorted(committed_ids - generated_ids)
    extra = sorted(generated_ids - committed_ids)

    require(
        generated_ids == committed_ids,
        f"{label}: generated patient-ID set matches committed artifact",
        f"missing={missing[:5]}, extra={extra[:5]}",
    )


def main() -> int:
    root = project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mama-workbook",
        type=Path,
        default=root / "tables" / "clinical_and_imaging_info.xlsx",
    )
    parser.add_argument(
        "--development-roster",
        type=Path,
        default=root / "BreastDCEDL_ISPY2_metadata.csv",
    )
    parser.add_argument(
        "--committed-external",
        type=Path,
        default=root / "external_manifest" / "external_manifest.csv",
    )
    parser.add_argument(
        "--committed-removed",
        type=Path,
        default=root / "external_manifest" / "removed_overlap.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reproducibility" / "_generated",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run checks without writing generated CSV/JSON files.",
    )
    args = parser.parse_args()

    print("Leakage-controlled external cohort reconstruction")
    print("=" * 64)

    require(args.mama_workbook.exists(), "MAMA-MIA workbook exists", str(args.mama_workbook))
    require(
        args.development_roster.exists(),
        "BreastDCEDL/I-SPY2 development roster exists",
        str(args.development_roster),
    )

    mama, sheet = load_mama_table(args.mama_workbook)
    development = pd.read_csv(args.development_roster)

    # Normalize column names without altering the committed source files.
    mama.columns = [str(c).strip().lower() for c in mama.columns]
    development.columns = [str(c).strip() for c in development.columns]

    require("patient_id" in mama.columns, "MAMA-MIA table has patient_id")
    require("dataset" in mama.columns, "MAMA-MIA table has dataset")
    require("pid" in development.columns, "Development roster has pid")

    mama = mama.copy()
    development = development.copy()

    mama["_normalized_id"] = mama["patient_id"].map(normalize_mama_id)
    development["_normalized_id"] = development["pid"].map(normalize_development_id)

    require(
        mama["_normalized_id"].is_unique,
        "Normalized MAMA-MIA patient IDs are unique",
    )
    require(
        development["_normalized_id"].is_unique,
        "Normalized development patient IDs are unique",
    )

    dev_ids = set(development["_normalized_id"])
    mama["_overlap_with_development"] = mama["_normalized_id"].isin(dev_ids)

    removed = mama[mama["_overlap_with_development"]].copy()
    external = mama[~mama["_overlap_with_development"]].copy()

    dataset_norm = mama["dataset"].astype(str).str.strip().str.upper()
    ispy2_label = dataset_norm.eq("ISPY2")

    require(
        len(mama) == EXPECTED["mama_total"],
        f"Full MAMA-MIA count = {EXPECTED['mama_total']}",
        f"observed={len(mama)}, sheet={sheet}",
    )
    require(
        len(development) == EXPECTED["development_total"],
        f"Development roster count = {EXPECTED['development_total']}",
        f"observed={len(development)}",
    )
    require(
        len(removed) == EXPECTED["overlap"],
        f"Identifier-matched overlap count = {EXPECTED['overlap']}",
        f"observed={len(removed)}",
    )
    require(
        len(external) == EXPECTED["external_total"],
        f"Retained external count = {EXPECTED['external_total']}",
        f"observed={len(external)}",
    )

    # Strong audit: the ID-based matches should coincide exactly with the
    # MAMA-MIA rows labelled as I-SPY2 for this dataset version.
    require(
        mama["_overlap_with_development"].equals(ispy2_label),
        "ID-based overlap set equals MAMA-MIA I-SPY2-source rows",
        (
            f"id_matches={int(mama['_overlap_with_development'].sum())}, "
            f"dataset_ISPY2={int(ispy2_label.sum())}"
        ),
    )

    external_counts = (
        external["dataset"].astype(str).str.strip().str.upper().value_counts().to_dict()
    )
    for cohort, expected_n in EXPECTED["cohorts"].items():
        require(
            int(external_counts.get(cohort, 0)) == expected_n,
            f"{cohort} retained count = {expected_n}",
            f"observed={external_counts.get(cohort, 0)}",
        )

    compare_id_sets(external, args.committed_external, "External manifest")
    compare_id_sets(removed, args.committed_removed, "Removed-overlap manifest")

    summary = {
        "mama_workbook": str(args.mama_workbook),
        "mama_sheet": sheet,
        "development_roster": str(args.development_roster),
        "mama_total": int(len(mama)),
        "development_total": int(len(development)),
        "overlap_removed": int(len(removed)),
        "external_total": int(len(external)),
        "external_cohorts": {k: int(v) for k, v in sorted(external_counts.items())},
    }

    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        external.to_csv(args.output_dir / "external_manifest_rebuilt.csv", index=False)
        removed.to_csv(args.output_dir / "removed_overlap_rebuilt.csv", index=False)
        (args.output_dir / "cohort_construction_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[PASS] Generated audit outputs written to {args.output_dir}")

    print("=" * 64)
    print("ALL COHORT-CONSTRUCTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nCohort reconstruction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
