# run_15_final_lcc_collapse_profile.py
# Final audit of LCC-collapse, opposite-half flags, and review-packet membership.
# No command-line parameters. Edit CONFIG only if your folder names changed.

from pathlib import Path
import re
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ============================================================
# CONFIG — EDIT ONLY HERE IF NEEDED
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUT_ROOT = SCRIPT_DIR
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Prior analysis folders
PATH_08 = PROJECT_ROOT / "08_external_failure_attribution_analysis" / "metrics" / "failure_attribution_case_table_enhanced.csv"
PATH_10 = PROJECT_ROOT / "10_final_external_sanity_checks" / "metrics" / "final_sanity_case_table_with_volume_bounds.csv"
PATH_13 = PROJECT_ROOT / "13_final_lcc_side_audit" / "metrics" / "independent_lcc_side_audit_all_cases.csv"
PATH_14 = PROJECT_ROOT / "14_lcc_collapse_visual_pattern_review" / "metrics" / "all_case_independent_lcc_collapse_pattern_audit.csv"
PATH_14_REVIEW_PREVIEWS = PROJECT_ROOT / "14_lcc_collapse_visual_pattern_review" / "review_packet" / "previews"
PATH_14_REVIEW_NIFTI = PROJECT_ROOT / "14_lcc_collapse_visual_pattern_review" / "review_packet" / "nifti_cases"

# Definitions used in previous scripts. Keep fixed for reproducibility.
RAW_DICE_COLLAPSE_MIN = 0.20
LCC_DICE_COLLAPSE_MAX = 0.05
CENTROID_DISTANCE_MM_THRESHOLD = 100.0

# ============================================================
# HELPERS
# ============================================================

def read_csv_if_exists(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: missing {name}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {name}: {path}  shape={df.shape}")
    return df


def normalize_case_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = s.replace(".NII.GZ", "").replace(".NII", "")
    s = re.sub(r"_(0000|0001|0002|0003|0004)$", "", s)
    return s


def find_col(df: pd.DataFrame, candidates, contains_all=None, contains_any=None):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates or []:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if contains_all:
        toks = [t.lower() for t in contains_all]
        for c in cols:
            cl = c.lower()
            if all(t in cl for t in toks):
                return c
    if contains_any:
        toks = [t.lower() for t in contains_any]
        for c in cols:
            cl = c.lower()
            if any(t in cl for t in toks):
                return c
    return None


def to_bool_series(s):
    if s is None:
        return None
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def safe_float_series(df, col, default=np.nan):
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def infer_cohort(case_id: str) -> str:
    s = normalize_case_id(case_id)
    if s.startswith("DUKE"):
        return "DUKE"
    if s.startswith("ISPY1"):
        return "ISPY1"
    if s.startswith("NACT"):
        return "NACT"
    return "OTHER"


def merge_sources():
    # Prefer 14/13 for flags, 10/08 for enriched case features.
    dfs = []
    for name, path in [("d14", PATH_14), ("d13", PATH_13), ("d10", PATH_10), ("d08", PATH_08)]:
        df = read_csv_if_exists(path, name)
        if len(df) == 0:
            continue
        case_col = find_col(df, ["case_id_std", "case_id", "pid", "patient_id"], contains_any=["case"])
        if case_col is None:
            print(f"WARNING: no case column in {name}")
            continue
        df = df.copy()
        df["case_id_std"] = df[case_col].apply(normalize_case_id)
        # Prefix duplicate non-key columns with source only if needed.
        dfs.append((name, df))

    if not dfs:
        raise FileNotFoundError("No usable prior CSV tables found. Run analyses 08, 10, 13, 14 first.")

    # Start with the most complete/all-case table. Use 14 if available, else 13/10/08.
    base_name, base = dfs[0]
    out = base.copy()
    for name, df in dfs[1:]:
        add_cols = [c for c in df.columns if c != "case_id_std" and c not in out.columns]
        out = out.merge(df[["case_id_std"] + add_cols], on="case_id_std", how="left")
    return out


def ensure_core_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cohort" not in df.columns:
        df["cohort"] = df["case_id_std"].apply(infer_cohort)

    # Dice columns
    raw_dice_col = find_col(df, ["v2_raw_dice", "raw_dice", "independent_v2_raw_dice"], contains_all=["v2", "raw", "dice"])
    lcc_dice_col = find_col(df, ["v2_lcc_dice", "lcc_dice", "independent_v2_lcc_dice"], contains_all=["v2", "lcc", "dice"])
    if raw_dice_col is None or lcc_dice_col is None:
        raise KeyError("Could not find v2 raw/LCC Dice columns in merged table.")
    df["audit_v2_raw_dice"] = safe_float_series(df, raw_dice_col)
    df["audit_v2_lcc_dice"] = safe_float_series(df, lcc_dice_col)

    # Collapse flag: prefer independent flag, otherwise recompute.
    collapse_col = find_col(
        df,
        ["lcc_collapse", "is_lcc_collapse", "v2_lcc_collapse", "audit_lcc_collapse", "lcc_collapse_flag"],
        contains_all=["collapse"],
    )
    collapse_series = to_bool_series(df[collapse_col]) if collapse_col else None
    if collapse_series is None:
        collapse_series = (df["audit_v2_raw_dice"] > RAW_DICE_COLLAPSE_MIN) & (df["audit_v2_lcc_dice"] < LCC_DICE_COLLAPSE_MAX)
    df["audit_lcc_collapse"] = collapse_series.fillna(False)

    # Opposite-half flag: try many likely names. Do NOT derive from collapse.
    opp_col = find_col(
        df,
        [
            "opposite_half_flag",
            "lcc_opposite_half_flag",
            "audit_opposite_half_flag",
            "lcc_opposite_side_flag",
            "lcc_opposite_image_half_flag",
            "opposite_side_flag",
            "opposite_half",
        ],
        contains_all=["opposite"],
    )
    opp_series = to_bool_series(df[opp_col]) if opp_col else None
    if opp_series is None:
        print("WARNING: no opposite-half flag column found. Creating all-False placeholder.")
        opp_series = pd.Series(False, index=df.index)
    df["audit_opposite_half_flag"] = opp_series.fillna(False)

    # Centroid distance flag
    dist_col = find_col(
        df,
        ["lcc_centroid_distance_mm", "v2_lcc_centroid_distance_mm", "centroid_distance_mm", "lcc_gt_centroid_distance_mm"],
        contains_all=["centroid", "distance"],
    )
    df["audit_lcc_centroid_distance_mm"] = safe_float_series(df, dist_col)
    df["audit_lcc_centroid_gt100mm"] = df["audit_lcc_centroid_distance_mm"] > CENTROID_DISTANCE_MM_THRESHOLD

    # Volume and component feature candidates
    for canonical, candidates, contains in [
        ("gt_vol_cm3", ["gt_vol_cm3", "gt_volume_cm3"], ["gt", "vol"]),
        ("v2_raw_pred_vol_cm3", ["v2_raw_pred_vol_cm3", "raw_pred_vol_cm3"], ["v2", "raw", "pred", "vol"]),
        ("v2_lcc_pred_vol_cm3", ["v2_lcc_pred_vol_cm3", "lcc_pred_vol_cm3"], ["v2", "lcc", "pred", "vol"]),
        ("v2_raw_n_components", ["v2_raw_n_components", "raw_n_components"], ["v2", "raw", "components"]),
        ("v2_raw_largest_component_fraction", ["v2_raw_largest_component_fraction", "raw_largest_component_fraction"], ["v2", "raw", "largest", "fraction"]),
        ("v2_lcc_pred_to_gt_vol_ratio", ["v2_lcc_pred_to_gt_vol_ratio", "pred_to_gt_vol_ratio"], ["v2", "lcc", "ratio"]),
        ("v2_raw_pred_to_gt_vol_ratio", ["v2_raw_pred_to_gt_vol_ratio"], ["v2", "raw", "ratio"]),
        ("volume_bound_signature", ["v2_lcc_volume_bound_failure_signature", "volume_bound_failure_signature", "failure_signature"], ["signature"]),
        ("volume_bound_group", ["volume_bound_diagnostic_group", "v2_lcc_volume_bound_diagnostic_group"], ["volume", "group"]),
        ("volume_quintile", ["gt_volume_quintile", "volume_quintile", "gt_vol_quintile"], ["quintile"]),
    ]:
        col = find_col(df, candidates, contains_all=contains)
        if col and col in df.columns:
            df[canonical] = df[col]

    # Pattern groups for analysis
    df["collapse_and_opposite"] = df["audit_lcc_collapse"] & df["audit_opposite_half_flag"]
    df["collapse_not_opposite"] = df["audit_lcc_collapse"] & (~df["audit_opposite_half_flag"])
    df["opposite_not_collapse"] = (~df["audit_lcc_collapse"]) & df["audit_opposite_half_flag"]
    df["neither_collapse_nor_opposite"] = (~df["audit_lcc_collapse"]) & (~df["audit_opposite_half_flag"])

    def pattern(row):
        if row["collapse_and_opposite"]:
            return "collapse_and_opposite"
        if row["collapse_not_opposite"]:
            return "collapse_not_opposite"
        if row["opposite_not_collapse"]:
            return "opposite_not_collapse"
        return "neither"

    df["lcc_opposite_pattern"] = df.apply(pattern, axis=1)
    return df


def flag_set_audit(df: pd.DataFrame):
    collapse_ids = set(df.loc[df["audit_lcc_collapse"], "case_id_std"])
    opposite_ids = set(df.loc[df["audit_opposite_half_flag"], "case_id_std"])
    centroid_ids = set(df.loc[df["audit_lcc_centroid_gt100mm"], "case_id_std"])
    union_ids = collapse_ids | opposite_ids
    inter_ids = collapse_ids & opposite_ids

    # Review packet IDs: parse previews and NIfTI subfolders if present.
    preview_ids = set()
    if PATH_14_REVIEW_PREVIEWS.exists():
        for p in PATH_14_REVIEW_PREVIEWS.glob("*.png"):
            m = re.match(r"([A-Za-z0-9]+_[A-Za-z0-9]+)", p.stem)
            if m:
                preview_ids.add(normalize_case_id(m.group(1)))
    nifti_ids = set()
    if PATH_14_REVIEW_NIFTI.exists():
        for p in PATH_14_REVIEW_NIFTI.iterdir():
            if p.is_dir():
                nifti_ids.add(normalize_case_id(p.name))
    review_ids = preview_ids | nifti_ids

    summary_rows = []
    def add(metric, value):
        summary_rows.append({"metric": metric, "value": value})

    add("n_all_cases", len(df))
    add("n_lcc_collapse", len(collapse_ids))
    add("n_opposite_half", len(opposite_ids))
    add("n_centroid_gt_100mm", len(centroid_ids))
    add("n_collapse_intersect_opposite", len(inter_ids))
    add("n_collapse_not_opposite", len(collapse_ids - opposite_ids))
    add("n_opposite_not_collapse", len(opposite_ids - collapse_ids))
    add("collapse_set_equals_opposite_set", collapse_ids == opposite_ids)
    add("n_union_collapse_or_opposite", len(union_ids))
    add("n_review_packet_preview_ids", len(preview_ids))
    add("n_review_packet_nifti_ids", len(nifti_ids))
    add("n_review_packet_union_ids", len(review_ids))
    add("n_union_missing_from_review_packet", len(union_ids - review_ids))
    add("n_review_packet_extra_not_in_union", len(review_ids - union_ids))

    # 2x2 table and Fisher/chi-square
    a = len(collapse_ids & opposite_ids)
    b = len(collapse_ids - opposite_ids)
    c = len(opposite_ids - collapse_ids)
    d = len(set(df["case_id_std"]) - collapse_ids - opposite_ids)
    table = np.array([[a, b], [c, d]], dtype=int)
    odds_ratio = np.inf if b * c == 0 else (a * d) / (b * c)
    add("odds_ratio_lcc_collapse_vs_opposite_half", odds_ratio)
    if SCIPY_AVAILABLE:
        fisher_or, fisher_p = fisher_exact(table)
        chi2, chi2_p, _, _ = chi2_contingency(table)
        add("fisher_exact_odds_ratio", fisher_or)
        add("fisher_exact_p_value", fisher_p)
        add("chi_square_statistic", chi2)
        add("chi_square_p_value", chi2_p)

    pd.DataFrame(summary_rows).to_csv(METRICS_DIR / "set_intersection_and_review_packet_audit.csv", index=False)
    pd.DataFrame(table, index=["LCC-collapse", "not LCC-collapse"], columns=["opposite-half", "not opposite-half"]).to_csv(METRICS_DIR / "crosstab_lcc_collapse_vs_opposite_half.csv")

    # Case lists for exact audit
    def save_case_list(ids, filename):
        sub = df[df["case_id_std"].isin(ids)].copy().sort_values(["cohort", "case_id_std"])
        sub.to_csv(METRICS_DIR / filename, index=False)
        return sub

    save_case_list(collapse_ids, "case_list_lcc_collapse.csv")
    save_case_list(opposite_ids, "case_list_opposite_half.csv")
    save_case_list(collapse_ids & opposite_ids, "case_list_collapse_intersect_opposite.csv")
    save_case_list(collapse_ids - opposite_ids, "case_list_collapse_not_opposite.csv")
    save_case_list(opposite_ids - collapse_ids, "case_list_opposite_not_collapse.csv")
    save_case_list(union_ids - review_ids, "case_list_union_missing_from_review_packet.csv")
    save_case_list(review_ids - union_ids, "case_list_review_packet_extra_not_in_union.csv")

    return {
        "collapse_ids": collapse_ids,
        "opposite_ids": opposite_ids,
        "centroid_ids": centroid_ids,
        "union_ids": union_ids,
        "review_ids": review_ids,
        "table": table,
        "odds_ratio": odds_ratio,
        "fisher_p": float(fisher_p) if SCIPY_AVAILABLE else np.nan,
        "chi2_p": float(chi2_p) if SCIPY_AVAILABLE else np.nan,
    }


def profile_groups(df: pd.DataFrame):
    # Key features to profile. Missing columns are ignored.
    features = [
        "audit_v2_raw_dice",
        "audit_v2_lcc_dice",
        "audit_lcc_centroid_distance_mm",
        "gt_vol_cm3",
        "v2_raw_pred_vol_cm3",
        "v2_lcc_pred_vol_cm3",
        "v2_raw_pred_to_gt_vol_ratio",
        "v2_lcc_pred_to_gt_vol_ratio",
        "v2_raw_n_components",
        "v2_raw_largest_component_fraction",
    ]
    features = [f for f in features if f in df.columns]

    rows = []
    for group, grp in df.groupby("lcc_opposite_pattern", dropna=False):
        row = {"pattern_group": group, "n": len(grp)}
        for f in features:
            vals = pd.to_numeric(grp[f], errors="coerce")
            row[f + "_mean"] = vals.mean()
            row[f + "_median"] = vals.median()
            row[f + "_std"] = vals.std()
            row[f + "_q25"] = vals.quantile(0.25)
            row[f + "_q75"] = vals.quantile(0.75)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("pattern_group")
    summary.to_csv(METRICS_DIR / "profile_lcc_collapse_opposite_pattern_groups.csv", index=False)

    # Cohort breakdown
    cohort_ct = pd.crosstab(df["cohort"], df["lcc_opposite_pattern"])
    cohort_ct.to_csv(METRICS_DIR / "cohort_breakdown_lcc_opposite_pattern_groups.csv")

    # Failure-signature breakdown if present
    for col, filename in [
        ("volume_bound_signature", "signature_breakdown_lcc_opposite_pattern_groups.csv"),
        ("volume_bound_group", "volume_group_breakdown_lcc_opposite_pattern_groups.csv"),
        ("volume_quintile", "volume_quintile_breakdown_lcc_opposite_pattern_groups.csv"),
    ]:
        if col in df.columns:
            pd.crosstab(df["lcc_opposite_pattern"], df[col]).to_csv(METRICS_DIR / filename)

    return summary, cohort_ct


def make_figures(df: pd.DataFrame, audit):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping figures")
        return

    # 2x2 count bar-ish heatmap
    table = audit["table"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=180)
    im = ax.imshow(table, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["opposite-half", "not opposite"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LCC-collapse", "not collapse"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(table[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("LCC-collapse vs opposite-image-half flag")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_lcc_collapse_vs_opposite_half_crosstab.png")
    plt.close(fig)

    # Pattern counts by cohort
    ct = pd.crosstab(df["cohort"], df["lcc_opposite_pattern"])
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    bottom = np.zeros(len(ct.index))
    x = np.arange(len(ct.index))
    for col in ct.columns:
        vals = ct[col].values
        ax.bar(x, vals, bottom=bottom, label=col)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index)
    ax.set_ylabel("Cases")
    ax.set_title("LCC-collapse / opposite-half pattern groups by cohort")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_pattern_groups_by_cohort.png")
    plt.close(fig)

    # Raw vs LCC Dice by pattern group
    if "audit_v2_raw_dice" in df.columns and "audit_v2_lcc_dice" in df.columns:
        plot_df = df.copy()
        groups = list(plot_df["lcc_opposite_pattern"].dropna().unique())
        groups = sorted(groups)
        means_raw = [plot_df.loc[plot_df["lcc_opposite_pattern"] == g, "audit_v2_raw_dice"].mean() for g in groups]
        means_lcc = [plot_df.loc[plot_df["lcc_opposite_pattern"] == g, "audit_v2_lcc_dice"].mean() for g in groups]
        x = np.arange(len(groups))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
        ax.bar(x - width/2, means_raw, width, label="v2 raw")
        ax.bar(x + width/2, means_lcc, width, label="v2 + LCC")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Dice")
        ax.set_title("Raw vs LCC Dice by LCC-collapse/opposite-half group")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig03_raw_vs_lcc_dice_by_pattern_group.png")
        plt.close(fig)


def write_takeaways(df: pd.DataFrame, audit, profile: pd.DataFrame):
    a, b = audit["table"][0, 0], audit["table"][0, 1]
    c, d = audit["table"][1, 0], audit["table"][1, 1]
    n = len(df)
    collapse_n = a + b
    opp_n = a + c
    union_n = len(audit["union_ids"])
    review_n = len(audit["review_ids"])

    lines = []
    lines.append("15 FINAL LCC-COLLAPSE PROFILE — QUICK TAKEAWAYS")
    lines.append("=" * 72)
    lines.append(f"Cases analyzed: {n}")
    lines.append("")
    lines.append("Set counts:")
    lines.append(f"  LCC-collapse: {collapse_n} / {n} ({100*collapse_n/n:.1f}%)")
    lines.append(f"  Opposite-image-half flags: {opp_n} / {n} ({100*opp_n/n:.1f}%)")
    lines.append(f"  Collapse ∩ opposite: {a} cases")
    lines.append(f"  Collapse outside opposite: {b} cases")
    lines.append(f"  Opposite outside collapse: {c} cases")
    lines.append(f"  Neither: {d} cases")
    lines.append(f"  Union(collapse, opposite): {union_n} cases")
    lines.append(f"  Review-packet IDs parsed: {review_n} cases")
    lines.append(f"  Union missing from review packet: {len(audit['union_ids'] - audit['review_ids'])}")
    lines.append(f"  Review packet extra not in union: {len(audit['review_ids'] - audit['union_ids'])}")
    lines.append("")
    lines.append("Association statistics:")
    lines.append(f"  Odds ratio: {audit['odds_ratio']:.3g}")
    if not math.isnan(audit.get("fisher_p", np.nan)):
        lines.append(f"  Fisher exact p-value: {audit['fisher_p']:.3e}")
    if not math.isnan(audit.get("chi2_p", np.nan)):
        lines.append(f"  Chi-square p-value: {audit['chi2_p']:.3e}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Opposite-image-half / large spatial displacement is strongly associated")
    lines.append("  with LCC-collapse, but it is not identical to LCC-collapse. It accounts")
    lines.append(f"  for {a}/{collapse_n} collapse cases ({100*a/max(collapse_n,1):.1f}%).")
    lines.append("  The remaining collapse cases should be described as a separate failure")
    lines.append("  pattern, likely involving component-selection or fragmentation without")
    lines.append("  the opposite-half displacement pattern.")
    lines.append("")
    lines.append("Manuscript-safe wording:")
    lines.append("  'LCC postprocessing improved cohort-average Dice but produced collapse-type")
    lines.append("  failures in a subset of external cases. Opposite-image-half or large spatial")
    lines.append("  displacement was strongly associated with these collapse failures, but did")
    lines.append("  not fully explain them, indicating heterogeneous postprocessing failure modes.'")
    lines.append("")
    lines.append("Important caution:")
    lines.append("  Do not write that side mismatch caused all collapse cases. Use 'associated")
    lines.append("  with' and 'spatially displaced/opposite-image-half component selection'.")

    out = OUT_ROOT / "article_takeaways_15.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Takeaways saved: {out}")


def main():
    print("15 Final LCC-collapse profile and review-packet audit")
    print("=" * 58)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root : {OUT_ROOT}")
    print()

    df = merge_sources()
    df = ensure_core_flags(df)
    df.to_csv(METRICS_DIR / "all_case_lcc_collapse_profile_table.csv", index=False)
    print(f"Saved all-case profile table: {METRICS_DIR / 'all_case_lcc_collapse_profile_table.csv'}")

    audit = flag_set_audit(df)
    profile, cohort_ct = profile_groups(df)
    make_figures(df, audit)
    write_takeaways(df, audit, profile)

    print()
    print("DONE — 15 profile audit complete.")
    print(f"Metrics : {METRICS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Takeaways: {OUT_ROOT / 'article_takeaways_15.txt'}")
    print()
    print("Quick results:")
    print(f"  all cases: {len(df)}")
    print(f"  LCC-collapse: {int(df['audit_lcc_collapse'].sum())}")
    print(f"  opposite-half: {int(df['audit_opposite_half_flag'].sum())}")
    print(f"  collapse ∩ opposite: {int((df['audit_lcc_collapse'] & df['audit_opposite_half_flag']).sum())}")
    print(f"  collapse outside opposite: {int((df['audit_lcc_collapse'] & ~df['audit_opposite_half_flag']).sum())}")
    print(f"  opposite outside collapse: {int((~df['audit_lcc_collapse'] & df['audit_opposite_half_flag']).sum())}")
    print(f"  OR: {audit['odds_ratio']:.3g}")
    if not math.isnan(audit.get("fisher_p", np.nan)):
        print(f"  Fisher p: {audit['fisher_p']:.3e}")
    print(f"  union(collapse, opposite): {len(audit['union_ids'])}")
    print(f"  review packet IDs parsed: {len(audit['review_ids'])}")
    print(f"  missing from review packet: {len(audit['union_ids'] - audit['review_ids'])}")
    print(f"  extra in review packet: {len(audit['review_ids'] - audit['union_ids'])}")
    print()
    print("Next files to inspect:")
    print(f"  {METRICS_DIR / 'profile_lcc_collapse_opposite_pattern_groups.csv'}")
    print(f"  {METRICS_DIR / 'case_list_collapse_not_opposite.csv'}")
    print(f"  {METRICS_DIR / 'case_list_collapse_intersect_opposite.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR")
        print("===========")
        print(repr(e))
        print("If the all-case table was saved, send the console output and relevant CSVs.")
        raise
