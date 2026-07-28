# 15 Final LCC-collapse profile and review-packet audit

Run from inside this folder:

```powershell
python run_15_final_lcc_collapse_profile.py
```

This script does not recompute MRI masks. It merges saved CSV outputs from previous analyses and checks:

1. Exact set relationships between LCC-collapse and opposite-image-half flags.
2. Whether the review packet contains the expected union of collapse/opposite cases.
3. Cohort and feature profiles for:
   - collapse and opposite
   - collapse but not opposite
   - opposite but not collapse
   - neither
4. Fisher exact / chi-square association between LCC-collapse and opposite-half flag.
5. Tables and figures for manuscript/supplementary use.

Main outputs:

```text
metrics/all_case_lcc_collapse_profile_table.csv
metrics/set_intersection_and_review_packet_audit.csv
metrics/crosstab_lcc_collapse_vs_opposite_half.csv
metrics/profile_lcc_collapse_opposite_pattern_groups.csv
metrics/cohort_breakdown_lcc_opposite_pattern_groups.csv
metrics/case_list_collapse_intersect_opposite.csv
metrics/case_list_collapse_not_opposite.csv
metrics/case_list_opposite_not_collapse.csv
article_takeaways_15.txt
```

Safe wording for manuscript:

> LCC postprocessing improved cohort-average Dice but introduced collapse-type failures in a subset of external cases. Opposite-image-half or large spatial displacement was strongly associated with LCC-collapse, but did not fully explain it, indicating heterogeneous postprocessing failure mechanisms.

Do not write that side mismatch caused all collapse cases.
