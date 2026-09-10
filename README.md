# Cross-Cohort Reliability and Failure Attribution in Breast DCE-MRI Tumor Segmentation

This repository contains code and derived analysis artifacts supporting the external-validation and failure-analysis components of the study:

**Cross-Cohort Reliability and Failure Attribution in AI-Based Breast DCE-MRI Tumor Segmentation**

The repository is intended as a reproducibility and audit resource. It is **not** clinical software and is not intended for diagnosis, treatment, or clinical decision support.

## Reviewer quick start

From the repository root:

```bash
python -m pip install -r reproducibility/requirements.txt
python reproducibility/build_external_cohort.py
python reproducibility/verify_paper_numbers.py
```

`build_external_cohort.py` independently reconstructs the leakage-controlled external cohort by normalizing and comparing the MAMA-MIA I-SPY2 identifiers with the BreastDCEDL/I-SPY2 development roster. It verifies:

- MAMA-MIA cases: 1,506
- BreastDCEDL/I-SPY2 development cases: 982
- overlapping MAMA-MIA I-SPY2-source cases excluded: 980
- retained external cases: 526
- Duke: 291
- I-SPY1: 171
- NACT-Pilot: 64

The generated audit files are written to `reproducibility/_generated/` and are ignored by Git.

`verify_paper_numbers.py` checks the principal external-validation and LCC-failure values reported in the revised manuscript against the committed canonical CSV outputs.

## Study-to-code map

| Study stage | Reviewer-facing code / artifact | Purpose |
|---|---|---|
| Leakage-controlled cohort construction | `reproducibility/build_external_cohort.py` | Normalizes I-SPY2 identifiers, excludes overlapping MAMA-MIA I-SPY2-source cases, and verifies the 526-case external cohort |
| External phase selection / preprocessing | `06_external_validation/preprocess_mama_mia_nnunet_v2.py` | Timing-aware phase selection and five-channel external input construction |
| External evaluation | `06_external_validation/evaluate_external_validation.py` | Computes external segmentation metrics |
| Cohort/timing summaries | `06_external_validation/analyze_external_phase_subset_home.py` | Produces overall, cohort-specific, and timing-quality summaries |
| LCC postprocessing | `06_external_validation/postprocess_external_predictions.py` | Largest connected-component postprocessing |
| Failure attribution | `08_external_failure_attribution_analysis/run_08_failure_attribution_analysis.py` | Exploratory patient-level failure attribution |
| Canonical LCC-collapse audit | `15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/run_15b_canonical_lcc_flag_repair.py` | Recomputes the final LCC-collapse and opposite-image-half flags from masks |
| Manuscript traceability | `MANUSCRIPT_NUMBERS.md` | Maps reported values to source files and scripts |
| Lightweight verification | `reproducibility/verify_paper_numbers.py` | Checks key manuscript values directly from committed outputs |

## Canonical external-analysis outputs

The principal external summary files used for the revised manuscript are:

```text
external_manifest/external_phase_subset_summary_overall.csv
external_manifest/external_phase_subset_summary_by_cohort.csv
external_manifest/external_phase_subset_summary_by_overall_phase_quality.csv
15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/
  article_takeaways_15b.txt
  metrics/canonical_all_case_lcc_opposite_audit.csv
  metrics/canonical_crosstab_lcc_collapse_vs_opposite_half.csv
```

### Important note on historical audit folders

Folders `13_final_lcc_side_audit`, `14_lcc_collapse_visual_pattern_review`, and the original files in `15_final_lcc_collapse_profile` document exploratory audit stages and are retained for provenance.

**For the revised manuscript, the canonical LCC-collapse/opposite-half results are those in `15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/`.**

The canonical definitions are:

- **LCC-collapse:** raw Dice > 0.20 and LCC Dice < 0.05.
- **Opposite-image-half flag:** ground-truth and LCC centroids lie on opposite image halves with x-axis separation >= 30 mm.

The 30-mm separation is an analysis parameter used for the spatial-displacement audit; it is not a clinically validated threshold.

## Data access

Patient-level MRI data are not redistributed in this repository.

The study uses public research datasets described in the manuscript. Users should obtain the imaging data from the original dataset providers and comply with their access and attribution terms.

For MAMA-MIA downloads, `download_mama_mia.py` reads a Synapse personal access token from the environment variable `SYNAPSE_AUTH_TOKEN`; no credential is stored in source code.

PowerShell example:

```powershell
$env:SYNAPSE_AUTH_TOKEN="YOUR_TOKEN"
$env:MAMA_MIA_IMAGES_DIR="C:\path\to\mama_mia_images"
python download_mama_mia.py
```

Linux/macOS example:

```bash
export SYNAPSE_AUTH_TOKEN="YOUR_TOKEN"
export MAMA_MIA_IMAGES_DIR="/path/to/mama_mia_images"
python download_mama_mia.py
```

## Reproducibility scope

The repository retains the original research scripts and derived outputs so that the analysis history remains inspectable. Some computationally intensive steps require the original MRI data, trained model artifacts, and an nnU-Net-compatible environment. The scripts in `reproducibility/` are intentionally lightweight and focus on the reviewer-requested cohort-overlap audit and verification of the principal reported external-analysis numbers.

See `MANUSCRIPT_NUMBERS.md` for a direct manuscript-to-artifact map.
