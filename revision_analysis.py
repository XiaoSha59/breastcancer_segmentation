from pathlib import Path
import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import fisher_exact, chi2_contingency

# ============================================================
# PATHS — matched to your MRI Project
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\olegk\Desktop\MRI Project")
EXTERNAL_METRICS = PROJECT_ROOT / "external_manifest" / "external_phase_subset_case_metrics.csv"
GT_DIR = PROJECT_ROOT / "segmentations_2" / "expert"
PRED_V2_DIR = Path(r"C:\nnw\mama_mia_output_v2")
OUT_DIR = PROJECT_ROOT / "revision_analysis"

# Primary definitions used in the manuscript
RAW_COLLAPSE_THR = 0.20
LCC_COLLAPSE_THR = 0.05

# Sensitivity grids
RAW_THR_GRID = [0.15, 0.20, 0.25, 0.30]
LCC_THR_GRID = [0.01, 0.025, 0.05, 0.075, 0.10]
OPPOSITE_MM_GRID = [20.0, 30.0, 40.0, 50.0]


def norm_id(x):
    return str(x).strip().replace('.nii.gz', '').replace('.nii', '').upper()


def find_nifti(folder, case_id):
    cid = norm_id(case_id)
    for p in [
        folder / f"{cid}.nii.gz",
        folder / f"{cid.lower()}.nii.gz",
        folder / f"{cid}.nii",
        folder / f"{cid.lower()}.nii",
    ]:
        if p.exists():
            return p
    return None


def read_mask(path):
    img = sitk.ReadImage(str(path))
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    return arr, img


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    den = int(a.sum() + b.sum())
    return 1.0 if den == 0 else float(2 * np.logical_and(a, b).sum() / den)


def largest_cc(mask):
    if mask.sum() == 0:
        return mask.astype(np.uint8)
    lab, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == int(np.argmax(counts))).astype(np.uint8)


def centroid_zyx(mask):
    pts = np.argwhere(mask > 0)
    return None if pts.size == 0 else pts.mean(axis=0)


def opposite_geometry(gt, lcc, spacing_x):
    c_gt = centroid_zyx(gt)
    c_lcc = centroid_zyx(lcc)
    if c_gt is None or c_lcc is None:
        return False, np.nan
    mid_x = (gt.shape[2] - 1) / 2.0
    opposite = bool((c_gt[2] - mid_x) * (c_lcc[2] - mid_x) < 0)
    x_dist_mm = float(abs(c_gt[2] - c_lcc[2]) * spacing_x)
    return opposite, x_dist_mm


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, ctr-half), min(1, ctr+half)


def or_ci(a, b, c, d):
    aa, bb, cc, dd = map(float, [a, b, c, d])
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5; bb += 0.5; cc += 0.5; dd += 0.5
    OR = aa*dd/(bb*cc)
    se = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    lo = math.exp(math.log(OR) - 1.959963984540054*se)
    hi = math.exp(math.log(OR) + 1.959963984540054*se)
    return OR, lo, hi


def load_metrics():
    if not EXTERNAL_METRICS.exists():
        raise FileNotFoundError(EXTERNAL_METRICS)
    df = pd.read_csv(EXTERNAL_METRICS)
    need = ['case_id','cohort','v1_raw_dice','v1_lcc_dice','v2_raw_dice','v2_lcc_dice']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df['case_id'] = df['case_id'].map(norm_id)
    df['cohort'] = df['cohort'].astype(str).str.upper().str.strip()
    for c in ['v1_raw_dice','v1_lcc_dice','v2_raw_dice','v2_lcc_dice']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['collapse'] = (df.v2_raw_dice > RAW_COLLAPSE_THR) & (df.v2_lcc_dice < LCC_COLLAPSE_THR)
    df['v2_lcc_delta'] = df.v2_lcc_dice - df.v2_raw_dice
    return df


def comment4_sensitivity(df):
    raw_rows = []
    for t in RAW_THR_GRID:
        f = (df.v2_raw_dice > t) & (df.v2_lcc_dice < LCC_COLLAPSE_THR)
        vals = df.loc[f, 'v2_lcc_dice'].to_numpy(float)
        raw_rows.append({
            'raw_threshold': t,
            'lcc_threshold': LCC_COLLAPSE_THR,
            'n_cases': int(f.sum()),
            'max_post_lcc_dice': float(vals.max()) if len(vals) else np.nan,
            'n_post_lcc_exact_zero': int((vals == 0.0).sum()),
        })
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(OUT_DIR/'comment4_raw_threshold_sensitivity.csv', index=False)

    lcc_rows = []
    for t in LCC_THR_GRID:
        f = (df.v2_raw_dice > RAW_COLLAPSE_THR) & (df.v2_lcc_dice < t)
        lcc_rows.append({'raw_threshold': RAW_COLLAPSE_THR, 'lcc_threshold': t, 'n_cases': int(f.sum())})
    lcc = pd.DataFrame(lcc_rows)
    lcc.to_csv(OUT_DIR/'comment4_lcc_threshold_sensitivity.csv', index=False)
    return raw, lcc


def recompute_geometry(df):
    if not GT_DIR.exists():
        raise FileNotFoundError(GT_DIR)
    if not PRED_V2_DIR.exists():
        raise FileNotFoundError(PRED_V2_DIR)

    rows = []
    print(f"Recomputing geometry for {len(df)} external cases...")
    for i, r in df.reset_index(drop=True).iterrows():
        cid = r.case_id
        if (i+1) % 50 == 0 or i == 0 or i+1 == len(df):
            print(f"  {i+1}/{len(df)}  {cid}")

        gt_p = find_nifti(GT_DIR, cid)
        pr_p = find_nifti(PRED_V2_DIR, cid)
        if gt_p is None or pr_p is None:
            raise FileNotFoundError(f"Missing GT/prediction for {cid}: GT={gt_p}, pred={pr_p}")

        gt, gt_img = read_mask(gt_p)
        raw, _ = read_mask(pr_p)
        if gt.shape != raw.shape:
            raise RuntimeError(f"Shape mismatch {cid}: GT={gt.shape}, pred={raw.shape}")
        lcc = largest_cc(raw)
        raw_d = dice(raw, gt)
        lcc_d = dice(lcc, gt)
        pure_opp, x_mm = opposite_geometry(gt, lcc, float(gt_img.GetSpacing()[0]))

        rows.append({
            'case_id': cid,
            'cohort': r.cohort,
            'raw_dice_recomputed': raw_d,
            'lcc_dice_recomputed': lcc_d,
            'collapse_recomputed': bool(raw_d > RAW_COLLAPSE_THR and lcc_d < LCC_COLLAPSE_THR),
            'pure_opposite_halves': pure_opp,
            'x_distance_mm': x_mm,
        })

    g = pd.DataFrame(rows)
    for t in OPPOSITE_MM_GRID:
        g[f'opposite_ge_{int(t)}mm'] = g.pure_opposite_halves & (g.x_distance_mm >= t)
    g.to_csv(OUT_DIR/'comment6_case_level_threshold_audit.csv', index=False)
    return g


def comment6_sensitivity(g):
    collapse = g.collapse_recomputed.astype(bool)
    rows = []
    for t in OPPOSITE_MM_GRID:
        flag = g[f'opposite_ge_{int(t)}mm'].astype(bool)
        a = int((collapse & flag).sum())
        b = int((collapse & ~flag).sum())
        c = int((~collapse & flag).sum())
        d = int((~collapse & ~flag).sum())
        fisher_or, p = fisher_exact([[a,b],[c,d]])
        OR, lo, hi = or_ci(a,b,c,d)
        rows.append({
            'threshold_mm': t,
            'flagged_total': int(flag.sum()),
            'collapse_flagged': a,
            'collapse_not_flagged': b,
            'noncollapse_flagged': c,
            'noncollapse_not_flagged': d,
            'odds_ratio': float(fisher_or),
            'or_95ci_low': lo,
            'or_95ci_high': hi,
            'fisher_p': float(p),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR/'comment6_opposite_half_threshold_sensitivity.csv', index=False)
    return out


def comment7_summary(df, g):
    order = ['DUKE','ISPY1','NACT']
    rows = []
    for cohort in order + ['ALL']:
        sub = df if cohort == 'ALL' else df[df.cohort == cohort]
        n = len(sub)
        k = int(sub.collapse.sum())
        lo, hi = wilson(k, n)
        delta = sub.v2_lcc_delta
        rows.append({
            'cohort': cohort,
            'n': n,
            'baseline_raw_dice': float(sub.v1_raw_dice.mean()),
            'baseline_lcc_dice': float(sub.v1_lcc_dice.mean()),
            'timing_raw_dice': float(sub.v2_raw_dice.mean()),
            'timing_lcc_dice': float(sub.v2_lcc_dice.mean()),
            'timing_lcc_minus_raw': float(delta.mean()),
            'lcc_improved_n': int((delta > 1e-12).sum()),
            'lcc_worsened_n': int((delta < -1e-12).sum()),
            'lcc_unchanged_n': int((abs(delta) <= 1e-12).sum()),
            'collapse_n': k,
            'collapse_rate': k/n,
            'collapse_rate_95ci_low': lo,
            'collapse_rate_95ci_high': hi,
        })
    s = pd.DataFrame(rows)

    # Add 30-mm opposite-half counts.
    opp_rows = []
    for cohort in order + ['ALL']:
        sub = g if cohort == 'ALL' else g[g.cohort == cohort]
        flag = sub.opposite_ge_30mm.astype(bool)
        coll = sub.collapse_recomputed.astype(bool)
        opp_rows.append({
            'cohort': cohort,
            'opposite_30mm_n': int(flag.sum()),
            'collapse_and_opposite_30mm_n': int((coll & flag).sum()),
        })
    s = s.merge(pd.DataFrame(opp_rows), on='cohort', how='left')
    s.to_csv(OUT_DIR/'comment7_cohort_summary.csv', index=False)

    # 3x2 heterogeneity test for collapse rates.
    table = []
    cont_rows = []
    for cohort in order:
        sub = df[df.cohort == cohort]
        k = int(sub.collapse.sum())
        nk = len(sub) - k
        table.append([k, nk])
        cont_rows.append({'cohort':cohort,'collapse':k,'noncollapse':nk,'total':len(sub)})
    chi2, p, dof, expected = chi2_contingency(np.asarray(table))
    pd.DataFrame(cont_rows).to_csv(OUT_DIR/'comment7_collapse_contingency.csv', index=False)
    test = pd.DataFrame([{
        'test':'Pearson chi-square, cohort x collapse status',
        'chi2':float(chi2), 'df':int(dof), 'p_value':float(p),
        'min_expected_cell':float(np.min(expected)),
    }])
    test.to_csv(OUT_DIR/'comment7_collapse_heterogeneity_test.csv', index=False)

    # LaTeX-ready rows.
    names = {'DUKE':'Duke','ISPY1':'I-SPY1','NACT':'NACT-Pilot','ALL':'All'}
    lines = ['% Cohort & N & Baseline raw & Baseline+LCC & Timing raw & Timing+LCC & Collapse \\\\']
    for _, r in s.iterrows():
        lines.append(
            f"{names[r.cohort]} & {int(r.n)} & {r.baseline_raw_dice:.3f} & {r.baseline_lcc_dice:.3f} & "
            f"{r.timing_raw_dice:.3f} & {r.timing_lcc_dice:.3f} & "
            f"{int(r.collapse_n)}/{int(r.n)} ({100*r.collapse_rate:.1f}\\%) \\\\"
        )
    (OUT_DIR/'comment7_latex_table_rows.txt').write_text('\n'.join(lines), encoding='utf-8')
    return s, test


def validate(df, g, sens30):
    assert len(df) == 526, f"Expected 526 cases, got {len(df)}"
    counts = df.cohort.value_counts().to_dict()
    assert counts.get('DUKE') == 291 and counts.get('ISPY1') == 171 and counts.get('NACT') == 64, counts
    assert int(df.collapse.sum()) == 43
    assert int(df[df.cohort=='DUKE'].collapse.sum()) == 18
    assert int(df[df.cohort=='ISPY1'].collapse.sum()) == 20
    assert int(df[df.cohort=='NACT'].collapse.sum()) == 5
    assert (df.loc[df.collapse, 'v2_lcc_dice'].to_numpy() == 0.0).all()

    m = df[['case_id','collapse']].merge(g[['case_id','collapse_recomputed']], on='case_id')
    assert (m.collapse.astype(bool).to_numpy() == m.collapse_recomputed.astype(bool).to_numpy()).all()

    r = sens30.loc[np.isclose(sens30.threshold_mm, 30.0)].iloc[0]
    assert int(r.flagged_total) == 54
    assert int(r.collapse_flagged) == 30
    assert int(r.collapse_not_flagged) == 13
    assert int(r.noncollapse_flagged) == 24
    assert int(r.noncollapse_not_flagged) == 459
    assert np.isclose(float(r.odds_ratio), 44.13461538461539)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('OUTPUT_DIR  :', OUT_DIR)

    df = load_metrics()
    raw_s, lcc_s = comment4_sensitivity(df)
    g = recompute_geometry(df)
    c6 = comment6_sensitivity(g)
    c7, het = comment7_summary(df, g)
    validate(df, g, c6)

    # Human-readable report.
    report = []
    report.append('VALIDATION: PASS')
    report.append('N=526; Duke=291; I-SPY1=171; NACT=64')
    report.append('Collapse=43; Duke=18; I-SPY1=20; NACT=5')
    report.append('All 43 collapse cases have post-LCC Dice = 0.000')
    report.append('30-mm canonical: flagged=54; 30/43 collapse flagged; 24/483 non-collapse flagged; OR=44.134615')
    report.append('\nCOMMENT 4 RAW THRESHOLD SENSITIVITY\n' + raw_s.to_string(index=False))
    report.append('\nCOMMENT 4 LCC THRESHOLD SENSITIVITY\n' + lcc_s.to_string(index=False))
    report.append('\nCOMMENT 6 OPPOSITE-HALF THRESHOLD SENSITIVITY\n' + c6.to_string(index=False))
    report.append('\nCOMMENT 7 COHORT SUMMARY\n' + c7.to_string(index=False))
    report.append('\nCOMMENT 7 COLLAPSE HETEROGENEITY\n' + het.to_string(index=False))
    (OUT_DIR/'revision_analysis_report.txt').write_text('\n'.join(report), encoding='utf-8')

    print('\nDONE — all validation checks passed.')
    print('Please send me this file after the run:')
    print(OUT_DIR/'revision_analysis_report.txt')


if __name__ == '__main__':
    main()
