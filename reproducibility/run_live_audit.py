#!/usr/bin/env python3
"""Stage-by-stage live reproducibility audit for the web demo.

This script performs real checks against committed derived artifacts.
It intentionally does not retrain models or rerun GPU inference.

Lines beginning with @@STAGE are machine-readable progress markers consumed
by the local FastAPI backend; normal stdout is streamed to the browser.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


# Make stdout safe and consistent on Windows when the script is executed
# through a subprocess pipe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]

OVERALL = ROOT / "external_manifest" / "external_phase_subset_summary_overall.csv"
BY_COHORT = ROOT / "external_manifest" / "external_phase_subset_summary_by_cohort.csv"
BY_PHASE = ROOT / "external_manifest" / "external_phase_subset_summary_by_overall_phase_quality.csv"
CANONICAL = (
    ROOT
    / "15_final_lcc_collapse_profile"
    / "15b_canonical_lcc_flag_repair"
    / "metrics"
    / "canonical_all_case_lcc_opposite_audit.csv"
)


def stage(index: int, title: str) -> None:
    print(f"@@STAGE|{index}|{title}", flush=True)


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        log(f"[PASS] {label}")
        return
    suffix = f" -- {detail}" if detail else ""
    log(f"[FAIL] {label}{suffix}")
    raise AssertionError(label)


def rounded3(value: object) -> str:
    return f"{float(value):.3f}"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def run_child(script: Path, *args: str) -> None:
    command = [sys.executable, "-u", str(script), *args]
    log(f"[INFO] Running {script.relative_to(ROOT)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\r\n"), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{script.relative_to(ROOT)} exited with code {return_code}"
        )


def main() -> int:
    log("[INFO] Breast DCE-MRI reproducibility audit")
    log("[INFO] Audit mode validates committed analysis artifacts; GPU inference is not rerun.")

    # 1. Cohort construction
    stage(0, "Cohort Construction")
    run_child(ROOT / "reproducibility" / "build_external_cohort.py", "--no-write")
    log("[PASS] Leakage-controlled external cohort reconstructed from source rosters.")

    # 2. Phase-selection audit
    stage(1, "Phase Selection")
    phase = pd.read_csv(BY_PHASE)
    phase["overall_norm"] = phase["overall"].astype(str).str.strip().str.lower()
    phase_counts = {
        str(row["overall_norm"]): int(row["n"])
        for _, row in phase.iterrows()
    }
    require(phase_counts.get("good") == 154, "Timing-quality group good = 154")
    require(phase_counts.get("acceptable") == 213, "Timing-quality group acceptable = 213")
    require(phase_counts.get("poor") == 146, "Timing-quality group poor = 146")
    require(phase_counts.get("no_times") == 13, "Cases without usable timing metadata = 13")
    require(
        154 + 213 + 146 + 13 == 526,
        "Timing-quality groups account for all 526 external cases",
    )
    log("[INFO] Usable timing metadata: 513 cases")
    log("[INFO] Acquisition-order fallback: 13 cases")

    # 3. External evaluation audit
    stage(2, "External Evaluation")
    overall = pd.read_csv(OVERALL).iloc[0]
    cohorts = pd.read_csv(BY_COHORT)
    cohorts["cohort_norm"] = cohorts["cohort"].astype(str).str.strip().str.upper()

    require(int(overall["n"]) == 526, "External evaluation n = 526")
    require(rounded3(overall["v1_raw_dice_mean"]) == "0.547", "Baseline raw Dice = 0.547")
    require(rounded3(overall["v2_raw_dice_mean"]) == "0.555", "Timing-aware raw Dice = 0.555")

    expected_cohort_raw = {
        "DUKE": (291, "0.566"),
        "ISPY1": (171, "0.536"),
        "NACT": (64, "0.558"),
    }
    for cohort, (expected_n, expected_dice) in expected_cohort_raw.items():
        row = cohorts[cohorts["cohort_norm"] == cohort].iloc[0]
        require(int(row["n"]) == expected_n, f"{cohort} n = {expected_n}")
        require(
            rounded3(row["v2_raw_dice_mean"]) == expected_dice,
            f"{cohort} timing-aware raw Dice = {expected_dice}",
        )

    # 4. LCC postprocessing audit
    stage(3, "LCC Postprocessing")
    require(rounded3(overall["v1_lcc_dice_mean"]) == "0.569", "Baseline + LCC Dice = 0.569")
    require(rounded3(overall["v2_lcc_dice_mean"]) == "0.574", "Timing-aware + LCC Dice = 0.574")

    expected_cohort_lcc = {
        "DUKE": "0.593",
        "ISPY1": "0.534",
        "NACT": "0.596",
    }
    for cohort, expected_dice in expected_cohort_lcc.items():
        row = cohorts[cohorts["cohort_norm"] == cohort].iloc[0]
        require(
            rounded3(row["v2_lcc_dice_mean"]) == expected_dice,
            f"{cohort} timing-aware + LCC Dice = {expected_dice}",
        )

    log(
        "[INFO] Mean Dice change from timing-aware raw to timing-aware + LCC: "
        f"{float(overall['v2_lcc_dice_mean']) - float(overall['v2_raw_dice_mean']):+.4f}"
    )

    # 5. Failure attribution
    stage(4, "Failure Attribution")
    canonical = pd.read_csv(CANONICAL).copy()
    canonical["collapse"] = as_bool(canonical["canonical_lcc_collapse"])
    canonical["opposite"] = as_bool(canonical["canonical_lcc_opposite_half"])
    canonical["cohort_norm"] = canonical["cohort"].astype(str).str.strip().str.upper()

    require(len(canonical) == 526, "Canonical audit contains 526 cases")

    collapse = canonical["collapse"]
    opposite = canonical["opposite"]
    a = int((collapse & opposite).sum())
    b = int((collapse & ~opposite).sum())
    c = int((~collapse & opposite).sum())
    d = int((~collapse & ~opposite).sum())

    require(int(collapse.sum()) == 43, "LCC-collapse cases = 43 / 526")
    require(int(opposite.sum()) == 54, "Opposite-image-half flags = 54 / 526")
    require(a == 30, "Collapse AND opposite-image-half = 30 / 43")
    require(c == 24, "Non-collapse AND opposite-image-half = 24 / 483")

    odds_ratio = (a * d) / (b * c)
    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    require(
        math.isclose(odds_ratio, 44.13461538461539, abs_tol=1e-10),
        "Odds ratio = 44.13",
        f"observed={odds_ratio}",
    )
    require(
        math.isclose(fisher_or, odds_ratio, abs_tol=1e-10),
        "Fisher odds ratio matches canonical odds ratio",
    )
    require(
        math.isclose(fisher_p, 3.947682212151622e-24, rel_tol=0, abs_tol=1e-30),
        "Fisher exact p = 3.95e-24",
        f"observed={fisher_p}",
    )

    collapse_by_cohort = {
        cohort: int(group["collapse"].sum())
        for cohort, group in canonical.groupby("cohort_norm")
    }
    require(collapse_by_cohort.get("DUKE") == 18, "Duke LCC-collapse = 18 / 291")
    require(collapse_by_cohort.get("ISPY1") == 20, "I-SPY1 LCC-collapse = 20 / 171")
    require(collapse_by_cohort.get("NACT") == 5, "NACT-Pilot LCC-collapse = 5 / 64")

    # 6. Final manuscript audit
    stage(5, "Canonical Audit")
    run_child(ROOT / "reproducibility" / "verify_paper_numbers.py")
    log("[PASS] End-to-end lightweight reproducibility audit completed.")
    print("@@AUDIT_DONE|pass", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[FAIL] Reproducibility audit stopped: {exc}")
        print("@@AUDIT_DONE|fail", flush=True)
        raise SystemExit(1)
