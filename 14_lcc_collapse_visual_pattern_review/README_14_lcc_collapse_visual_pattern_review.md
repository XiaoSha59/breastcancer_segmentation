
# 14 LCC-collapse visual pattern review

This folder performs the final visual-pattern review requested after step 13.
It is designed to answer:

1. Are LCC-collapse cases and opposite-image-half flags identical, or only associated?
2. Which cases are collapse+opposite, collapse-not-opposite, and opposite-not-collapse?
3. Are LCC-collapse failures concentrated in one cohort?
4. What do the cases look like visually?
5. Can we safely write that LCC improves average Dice but can cause patient-level collapse failures?

## How to run

Place this folder at:

```text
C:\Users\olegk\Desktop\MRI Project\14_lcc_collapse_visual_pattern_review
```

Then run:

```text
python run_14_lcc_collapse_visual_pattern_review.py
```

No command-line arguments are required.

## Main outputs

```text
metrics/all_case_independent_lcc_collapse_pattern_audit.csv
metrics/lcc_collapse_opposite_set_summary.csv
metrics/crosstab_lcc_collapse_vs_opposite_half.csv
metrics/crosstab_lcc_collapse_opposite_centroid100_by_cohort.csv
metrics/lcc_collapse_opposite_by_cohort.csv
metrics/manual_review_checklist_lcc_collapse_patterns.csv
figures/contact_sheet_lcc_collapse_visual_patterns.jpg
review_packet/previews/
review_packet/nifti_cases/
article_takeaways_14.txt
```

## How to use in the paper

Use careful wording:

- LCC-collapse and opposite-image-half/spatial displacement were strongly associated, not identical.
- Opposite-half displacement accounted for part of LCC-collapse cases.
- LCC improves cohort-average Dice but can be unsafe at patient level.
- Do not call this confirmed anatomical laterality mismatch unless manually confirmed.

## Suggested safe wording

"LCC postprocessing improved mean external Dice but introduced a patient-level reliability trade-off. LCC-collapse cases were strongly enriched for spatially displaced or opposite-image-half component selection, but this pattern did not account for all collapse cases, indicating heterogeneous postprocessing failure mechanisms."
