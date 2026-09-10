#!/usr/bin/env python3
"""Verify principal external-analysis values reported in the revised manuscript.

This script reads committed derived outputs only. It does not require MRI images,
model weights, or GPU inference.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()

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

failures: list[str] = []


def pass_fail(label: str, ok: bool, observed: object, expected: object) -> None:
    if ok:
        print(f"[PASS] {label}: {observed}")
    else:
        msg = f"{label}: observed={observed}, expected={expected}"
        failures.append(msg)
        print(f"[FAIL] {msg}")


def check_int(label: str, observed: object, expected: int) -> None:
    try:
        obs = int(observed)
    except Exception:
        obs = observed
    pass_fail(label, obs == expected, obs, expected)


def check_float(
    label: str,
    observed: object,
    expected: float,
    tol: float = 5e-7,
    display: int | None = None,
) -> None:
    obs = float(observed)
    ok = math.isclose(obs, expected, rel_tol=0.0, abs_tol=tol)
    shown = f"{obs:.{display}f}" if display is not None else f"{obs:.12g}"
    exp_shown = f"{expected:.{display}f}" if display is not None else expected
    pass_fail(label, ok, shown, exp_shown)


def check_rounded3(label: str, observed: object, expected: str) -> None:
    obs = float(observed)
    shown = f"{obs:.3f}"
    pass_fail(label, shown == expected, shown, expected)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> int:
    print("Revised manuscript numerical verification")
    print("=" * 72)

    for path in [OVERALL, BY_COHORT, BY_PHASE, CANONICAL]:
        pass_fail(f"File exists: {path.relative_to(ROOT)}", path.exists(), path.exists(), True)
    if failures:
        return finish()

    overall = pd.read_csv(OVERALL)
    cohorts = pd.read_csv(BY_COHORT)
    phase = pd.read_csv(BY_PHASE)
    canonical = pd.read_csv(CANONICAL)

    # ---- Overall external metrics ----
    check_int("External n", overall.loc[0, "n"], 526)
    check_rounded3("Baseline raw Dice", overall.loc[0, "v1_raw_dice_mean"], "0.547")
    check_rounded3("Baseline LCC Dice", overall.loc[0, "v1_lcc_dice_mean"], "0.569")
    check_rounded3("Timing-aware raw Dice", overall.loc[0, "v2_raw_dice_mean"], "0.555")
    check_rounded3("Timing-aware + LCC Dice", overall.loc[0, "v2_lcc_dice_mean"], "0.574")

    # ---- Cohort-specific external metrics ----
    cohort_expect = {
        "DUKE": {
            "n": 291,
            "v1_raw_dice_mean": "0.554",
            "v1_lcc_dice_mean": "0.584",
            "v2_raw_dice_mean": "0.566",
            "v2_lcc_dice_mean": "0.593",
        },
        "ISPY1": {
            "n": 171,
            "v1_raw_dice_mean": "0.530",
            "v1_lcc_dice_mean": "0.528",
            "v2_raw_dice_mean": "0.536",
            "v2_lcc_dice_mean": "0.534",
        },
        "NACT": {
            "n": 64,
            "v1_raw_dice_mean": "0.562",
            "v1_lcc_dice_mean": "0.610",
            "v2_raw_dice_mean": "0.558",
            "v2_lcc_dice_mean": "0.596",
        },
    }

    cohorts["cohort_norm"] = cohorts["cohort"].astype(str).str.strip().str.upper()
    for cohort, exp in cohort_expect.items():
        rows = cohorts[cohorts["cohort_norm"] == cohort]
        check_int(f"{cohort} row count in summary", len(rows), 1)
        if len(rows) != 1:
            continue
        row = rows.iloc[0]
        check_int(f"{cohort} n", row["n"], exp["n"])
        for col in [
            "v1_raw_dice_mean",
            "v1_lcc_dice_mean",
            "v2_raw_dice_mean",
            "v2_lcc_dice_mean",
        ]:
            check_rounded3(f"{cohort} {col}", row[col], exp[col])

    # ---- Timing-quality groups ----
    phase["overall_norm"] = phase["overall"].astype(str).str.strip().str.lower()
    phase_expect = {
        "good": (154, "0.596"),
        "acceptable": (213, "0.558"),
        "poor": (146, "0.569"),
        "no_times": (13, "0.633"),
    }
    for group, (n_exp, dice_exp) in phase_expect.items():
        rows = phase[phase["overall_norm"] == group]
        check_int(f"Timing group {group}: row count", len(rows), 1)
        if len(rows) != 1:
            continue
        row = rows.iloc[0]
        check_int(f"Timing group {group}: n", row["n"], n_exp)
        check_rounded3(
            f"Timing group {group}: timing-aware + LCC Dice",
            row["v2_lcc_dice_mean"],
            dice_exp,
        )

    # ---- Canonical LCC audit ----
    required_cols = {
        "cohort",
        "canonical_lcc_collapse",
        "canonical_lcc_opposite_half",
    }
    missing_cols = required_cols - set(canonical.columns)
    pass_fail("Canonical audit required columns present", not missing_cols, sorted(required_cols), sorted(required_cols))
    if missing_cols:
        return finish()

    def as_bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.fillna(False)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(float) != 0
        return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})

    canonical = canonical.copy()
    canonical["collapse"] = as_bool(canonical["canonical_lcc_collapse"])
    canonical["opposite"] = as_bool(canonical["canonical_lcc_opposite_half"])
    canonical["cohort_norm"] = canonical["cohort"].astype(str).str.strip().str.upper()

    check_int("Canonical cases audited", len(canonical), 526)

    collapse_n = int(canonical["collapse"].sum())
    opposite_n = int(canonical["opposite"].sum())
    both_n = int((canonical["collapse"] & canonical["opposite"]).sum())
    collapse_not_opp = int((canonical["collapse"] & ~canonical["opposite"]).sum())
    noncollapse_opp = int((~canonical["collapse"] & canonical["opposite"]).sum())
    neither_n = int((~canonical["collapse"] & ~canonical["opposite"]).sum())

    check_int("LCC-collapse cases", collapse_n, 43)
    check_int("Opposite-image-half cases", opposite_n, 54)
    check_int("Collapse and opposite-half", both_n, 30)
    check_int("Collapse without opposite-half", collapse_not_opp, 13)
    check_int("Non-collapse and opposite-half", noncollapse_opp, 24)
    check_int("Neither collapse nor opposite-half", neither_n, 459)

    # 2x2 table:
    #                         opposite   not opposite
    # collapse                  a            b
    # non-collapse              c            d
    a, b, c, d = both_n, collapse_not_opp, noncollapse_opp, neither_n
    odds_ratio = (a * d) / (b * c)
    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low = math.exp(math.log(odds_ratio) - 1.96 * se_log_or)
    ci_high = math.exp(math.log(odds_ratio) + 1.96 * se_log_or)

    check_float("Odds ratio", odds_ratio, 44.13461538461539, tol=1e-10, display=2)
    check_float("Fisher odds ratio", fisher_or, 44.13461538461539, tol=1e-10, display=2)
    check_float("Fisher exact p", fisher_p, 3.947682212151622e-24, tol=1e-30)
    check_float("Odds-ratio CI lower", ci_low, 20.446891513774194, tol=1e-10, display=2)
    check_float("Odds-ratio CI upper", ci_high, 95.26456742022306, tol=1e-10, display=2)

    # Collapse by cohort
    collapse_by_cohort = {
        cohort: int(group["collapse"].sum())
        for cohort, group in canonical.groupby("cohort_norm")
    }
    for cohort, expected in {"DUKE": 18, "ISPY1": 20, "NACT": 5}.items():
        check_int(f"{cohort} LCC-collapse cases", collapse_by_cohort.get(cohort, 0), expected)

    # Wilson 95% CIs printed for reviewer traceability.
    print("\nCollapse-rate Wilson 95% CIs")
    for cohort, n, k in [
        ("DUKE", 291, 18),
        ("ISPY1", 171, 20),
        ("NACT", 64, 5),
        ("ALL", 526, 43),
    ]:
        lo, hi = wilson_interval(k, n)
        print(
            f"  {cohort:<6} {k}/{n} = {100*k/n:.1f}% "
            f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)"
        )

    return finish()


def finish() -> int:
    print("=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
