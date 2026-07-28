# 12 Final external case sanity review

This folder performs the final sanity checks before writing the external-validation/failure-attribution section.

It is designed to answer three reviewer-style questions:

1. Were the severe failures caused by a global geometry/orientation/pairing bug?
2. How often does LCC postprocessing catastrophically hurt cases where the raw prediction partly overlapped the tumor?
3. Do the large-centroid-distance failures suggest possible left/right side mismatch?

## How to run

Place this folder here:

```text
C:\Users\olegk\Desktop\MRI Project\12_final_external_case_sanity_review
```

Then run:

```text
python run_12_final_case_sanity_review.py
```

## Inputs expected

```text
C:\Users\olegk\Desktop\MRI Project\10_final_external_sanity_checks\metrics\final_sanity_case_table_with_volume_bounds.csv
C:\Users\olegk\Desktop\MRI Project\10_final_external_sanity_checks\metrics\predictor_intercorrelation_pairs.csv
C:\Users\olegk\Desktop\MRI Project\segmentations_2\expert
C:\nnw\mama_mia_output_v2
C:\nnw\mama_mia_input_v2
```

## Important outputs

```text
metrics\full_geometry_laterality_lcc_review_all_cases.csv
metrics\corrected_spatial_boundary_mismatch_cases.csv
metrics\lcc_collapse_cases_all_external.csv
metrics\opposite_side_suspect_cases.csv
metrics\large_centroid_distance_gt_100mm_cases.csv
metrics\lcc_tradeoff_summary_by_cohort.csv
metrics\cleaned_intercorrelations_no_self_or_timing_duplicates.csv
metrics\cleaned_intercorrelations_cross_family_only.csv
metrics\manual_review_checklist.csv
review_packet\previews\contact_sheet_selected_review_cases.jpg
review_packet\nifti_cases\CASE_ID\*.nii.gz
article_takeaways_12.txt
```

## How to interpret

- `any_geometry_mismatch = False` and recomputed Dice matching the prior tables means the evaluation is unlikely to be caused by a global spacing/orientation bug.
- `lcc_collapse_raw_gt_0p2_lcc_lt_0p05 = True` means raw prediction partly overlapped the tumor, but LCC reduced Dice to nearly zero. This is important evidence that LCC helps on average but can hurt individual cases.
- `lcc_opposite_side_suspect = True` means GT and LCC centroids lie on opposite halves of the image. This does **not** prove wrong patient pairing by itself; it flags cases for visual review.
- The manual review checklist can be edited after opening previews or 3D Slicer packets.

## Suggested manuscript wording

Use this as a targeted sanity review, not as a random-sample audit:

> Targeted review of severe external failures found no systematic spacing, orientation, or geometry mismatch between predictions and expert masks. Severe failures were characterized by large spatial displacement and component-selection errors; in some cases, largest-connected-component postprocessing removed tumor-overlapping components when the largest predicted component was not the true lesion. These findings support the interpretation that external degradation reflected genuine cross-domain segmentation instability rather than a global preprocessing artifact.
