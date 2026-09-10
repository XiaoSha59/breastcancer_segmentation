# Reproducibility helpers

These scripts are lightweight reviewer-facing checks. They do not retrain the segmentation model.

## 1. Install

From the repository root:

```bash
python -m pip install -r reproducibility/requirements.txt
```

## 2. Reconstruct the leakage-controlled external cohort

```bash
python reproducibility/build_external_cohort.py
```

The script reads:

```text
tables/clinical_and_imaging_info.xlsx
BreastDCEDL_ISPY2_metadata.csv
```

It normalizes the I-SPY2 identifiers, compares the two rosters, and verifies that 980 overlapping MAMA-MIA I-SPY2-source cases are excluded, leaving 526 external cases.

Generated audit copies are written to:

```text
reproducibility/_generated/
```

They are ignored by Git and do not replace the committed study outputs.

## 3. Verify principal manuscript numbers

```bash
python reproducibility/verify_paper_numbers.py
```

Expected final line:

```text
ALL CHECKS PASSED
```

If any check fails, do not update the manuscript or commit revised canonical results until the discrepancy is investigated.
