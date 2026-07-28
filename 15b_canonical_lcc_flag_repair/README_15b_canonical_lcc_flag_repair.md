# 15b Canonical LCC-collapse / opposite-half repair

This is a **supplemental script** to run inside:

```text
C:\Users\olegk\Desktop\MRI Project\15_final_lcc_collapse_profile
```

Run:

```text
python run_15b_canonical_lcc_flag_repair.py
```

It recomputes, from masks, one canonical definition for:

- LCC-collapse: raw Dice > 0.20 and LCC Dice < 0.05
- opposite-image-half flag: GT and LCC centroids on opposite image halves with enough x-distance
- centroid distance > 100 mm

It also compares D13/D14/D15 flags against this canonical result and rebuilds a corrected review packet from the **exact canonical union**:

```text
LCC-collapse OR opposite-image-half
```

Main outputs:

```text
15b_canonical_lcc_flag_repair/
  metrics/
    canonical_all_case_lcc_opposite_audit.csv
    canonical_lcc_opposite_intersection_summary.csv
    canonical_crosstab_lcc_collapse_vs_opposite_half.csv
    canonical_case_list_collapse_intersect_opposite.csv
    canonical_case_list_collapse_not_opposite.csv
    canonical_case_list_opposite_not_collapse.csv
    canonical_case_list_union_collapse_or_opposite.csv
    previous_audit_flag_set_differences_vs_canonical.csv
    canonical_review_packet_manifest.csv
  figures/
    contact_sheet_canonical_union_collapse_or_opposite.jpg
    fig01_canonical_pattern_group_counts.png
  review_packet/
    previews/
    nifti_cases/
  article_takeaways_15b.txt
```

Use the **canonical** numbers from this script for manuscript wording, not the inconsistent D13/D14/D15 counts.
