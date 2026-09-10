# Manuscript numbers and provenance

This file maps the principal revised-manuscript values from the external-validation and LCC-failure analyses to the committed result files and the scripts that generated them.

Values in the manuscript are generally rounded to three decimal places. The exact values below are retained so that rounding can be checked directly.

## Cohort construction

| Reported item | Value | Source / verification |
|---|---:|---|
| Full MAMA-MIA cohort | 1,506 | `tables/clinical_and_imaging_info.xlsx`; reconstructed by `reproducibility/build_external_cohort.py` |
| BreastDCEDL/I-SPY2 development roster | 982 | `BreastDCEDL_ISPY2_metadata.csv`; checked by `reproducibility/build_external_cohort.py` |
| MAMA-MIA I-SPY2-source cases excluded | 980 | `external_manifest/removed_overlap.csv`; reconstructed by `reproducibility/build_external_cohort.py` |
| External validation set | 526 | `external_manifest/external_manifest.csv` |
| Duke | 291 | `external_manifest/external_manifest.csv` |
| I-SPY1 | 171 | `external_manifest/external_manifest.csv` |
| NACT-Pilot | 64 | `external_manifest/external_manifest.csv` |

The overlap reconstruction normalizes BreastDCEDL IDs such as `ACRIN-6698-102212` to the MAMA-MIA form `ISPY2_102212` before set comparison. It then confirms that the 980 identifier matches correspond exactly to the MAMA-MIA rows labelled as I-SPY2.

## External segmentation performance

Primary source:
`external_manifest/external_phase_subset_summary_overall.csv`

Producer:
`06_external_validation/analyze_external_phase_subset_home.py`

| Setting | Exact mean Dice | Manuscript |
|---|---:|---:|
| Baseline phase selection, raw | 0.5473178721 | 0.547 |
| Baseline phase selection, LCC | 0.5687679087 | 0.569 |
| Timing-aware phase selection, raw | 0.5554280814 | 0.555 |
| Timing-aware phase selection, LCC | 0.5738137499 | 0.574 |

## External performance by source cohort

Primary source:
`external_manifest/external_phase_subset_summary_by_cohort.csv`

Producer:
`06_external_validation/analyze_external_phase_subset_home.py`

| Cohort | n | Baseline raw | Baseline LCC | Timing-aware raw | Timing-aware LCC |
|---|---:|---:|---:|---:|---:|
| Duke | 291 | 0.554 | 0.584 | 0.566 | 0.593 |
| I-SPY1 | 171 | 0.530 | 0.528 | 0.536 | 0.534 |
| NACT-Pilot | 64 | 0.562 | 0.610 | 0.558 | 0.596 |
| All | 526 | 0.547 | 0.569 | 0.555 | 0.574 |

The overall row is computed from unrounded case-level values; it can therefore differ slightly from a weighted average computed from the displayed three-decimal cohort means.

## Timing-quality groups

Primary source:
`external_manifest/external_phase_subset_summary_by_overall_phase_quality.csv`

| Timing group | n | Timing-aware + LCC mean Dice |
|---|---:|---:|
| good | 154 | 0.596 |
| acceptable | 213 | 0.558 |
| poor | 146 | 0.569 |
| no usable timing metadata | 13 | 0.633 |

The 13 `no_times` cases remain in the 526-case external analysis. In the timing-aware protocol they revert to acquisition-order phase selection.

## Canonical LCC-collapse / opposite-half audit

Canonical source:
`15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/metrics/canonical_all_case_lcc_opposite_audit.csv`

Canonical producer:
`15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/run_15b_canonical_lcc_flag_repair.py`

| Item | Value |
|---|---:|
| Cases audited | 526 |
| LCC-collapse | 43 / 526 (8.2%) |
| Opposite-image-half flag | 54 / 526 |
| Collapse and opposite-half | 30 / 43 |
| Non-collapse and opposite-half | 24 / 483 |
| Collapse without opposite-half | 13 |
| Non-collapse without opposite-half | 459 |
| Odds ratio | 44.13 |
| 95% CI for odds ratio | 20.45-95.26 |
| Fisher exact p-value | 3.948e-24 |

Canonical definitions:

- LCC-collapse: raw Dice > 0.20 and LCC Dice < 0.05.
- Opposite-image-half: GT and LCC centroids are on opposite image halves and are separated by at least 30 mm along x.

The opposite-half finding is an association and should not be interpreted as showing that spatial displacement caused every LCC-collapse event.

## LCC-collapse by cohort

These counts are recomputed directly from the canonical all-case audit by `reproducibility/verify_paper_numbers.py`.

| Cohort | Collapse | Rate |
|---|---:|---:|
| Duke | 18 / 291 | 6.2% |
| I-SPY1 | 20 / 171 | 11.7% |
| NACT-Pilot | 5 / 64 | 7.8% |
| All | 43 / 526 | 8.2% |

## Superseded exploratory values

The original file

`15_final_lcc_collapse_profile/article_takeaways_15.txt`

contains an earlier exploratory opposite-half audit. It is retained only for provenance and is **not** the source for the revised manuscript.

For the revised manuscript, always use the `15b_canonical_lcc_flag_repair` outputs listed above.
