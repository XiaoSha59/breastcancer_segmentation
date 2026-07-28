# 13 Final LCC / side-flag audit

This folder performs the final sanity checks requested after the 12th run:

1. Independently recomputes raw/LCC Dice from v2 prediction masks and expert masks.
2. Defines LCC-collapse over all 526 external cases: raw Dice > 0.20 but LCC Dice < 0.05.
3. Independently computes image-half/opposite-side flags for all cases, not only collapse cases.
4. Reports explicit set intersections:
   - LCC-collapse vs opposite-half flags
   - LCC-collapse vs centroid distance >100 mm
   - opposite-half flags vs centroid distance >100 mm
5. Reports cohort distribution of LCC-collapse/opposite/large-distance flags.
6. Adds a rough bilateral-FOV heuristic for flagged cases only.
7. Recomputes a stricter cleaned cross-family collinearity table.
8. Saves selected preview PNGs.

## Run

Place this folder here:

```text
C:\Users\olegk\Desktop\MRI Project\13_final_lcc_side_audit
```

Then run:

```text
python run_13_final_lcc_side_audit.py
```

## Key outputs

```text
metrics\independent_lcc_side_audit_all_cases.csv
metrics\flag_set_intersection_summary.csv
metrics\crosstab_lcc_collapse_vs_opposite_half.csv
metrics\crosstab_lcc_collapse_vs_centroid_gt100mm.csv
metrics\crosstab_3way_lcc_collapse_opposite_centroid100.csv
metrics\lcc_side_flags_by_cohort.csv
metrics\lcc_collapse_case_list.csv
metrics\opposite_half_case_list.csv
metrics\large_lcc_centroid_gt100mm_case_list.csv
metrics\cleaned_cross_family_predictor_correlations_strict.csv
figures\fig01_lcc_collapse_rate_by_cohort.png
figures\fig02_raw_vs_lcc_dice_collapse_highlight.png
previews_flagged_cases\*.png
article_takeaways_13.txt
```

## Interpretation warning

`opposite-half` is an image-half heuristic, not a confirmed anatomical laterality label.
Use wording like `large spatial displacement / possible wrong-component selection` unless manual 3D review confirms a true contralateral selection pattern.
