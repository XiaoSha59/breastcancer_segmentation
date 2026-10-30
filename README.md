# 3D Breast DCE-MRI Tumor Segmentation with Auxiliary HER2 Classification

An empirical controlled multi-task learning study exploring whether patient-level clinical supervision (HER2 status) via a shared encoder improves 3D volumetric tumor segmentation in dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI).

---

## 🔬 Research Overview

* **Primary Question (RQ1):** Holding the backbone, dataset, split, and training budget constant, does joint training with patient-level HER2 classification improve 3D tumor segmentation on Breast DCE-MRI compared to a segmentation-only baseline?
* **Development Cohort:** BreastDCEDL I-SPY2 ($n = 982$, split 784 train / 99 val / 99 test).
* **External Test Cohort:** I-SPY1 ($n = 171$, zero-shot evaluation).
* **Core Backbone:** 3D Residual Encoder nnU-Net (ResEnc M / ResEnc L).
* **Auxiliary Task:** Patch-gated weighted binary cross-entropy on bottleneck feature.

---

## 📁 Repository Structure

```text
breastcancer_segmentation/
├── configs/                     # Experiment configurations (B1 baseline, M1 MTL, C1 control)
├── data/                        # [Ignored] Extracted 3-phase NIfTI & manifests
├── outputs/                     # [Ignored] Checkpoints, predictions, and metrics
├── references/                  # Reference metadata, splits, and baseline scripts
│   ├── metadata/                # I-SPY2 & external manifests, clinical tables
│   └── krikun_code/             # Reference 5-channel & 200-epoch validation scripts
├── scripts/                     # Standalone CLI tools (extraction, QC, sanity tests)
├── src/                         # Core Python package
│   ├── data/                    # 5-channel dataset, patch sampling, and augmentations
│   ├── models/                  # Shared ResEnc encoder, decoder, classification head
│   ├── losses/                  # Patch-gated multi-task loss (DiceCE + WeightedBCE)
│   ├── train/                   # 200-epoch training engine with Poly LR schedule
│   └── evaluation/              # Evaluator (Dice, HD95, Paired Bootstrap, LCC audit)
├── .gitignore                   # Comprehensive ignore rules for data, weights, logs, reports
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Quick Start & Setup

### 1. Virtual Environment

Create and activate the virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛡️ Operational Principles

1. **Pre-flight Smoke Test:** Always run line-by-line review and overfit sanity check on 2 sample patients before initiating full-cohort training.
2. **Lossless Hardware Acceleration:** Use PyTorch AMP (`fp16`/`bf16`) and cuDNN benchmarking; no aggressive lossy compression.
3. **Auto-Shutdown on Complete/Failure:** Training jobs automatically persist emergency logs and trigger system shutdown when done or upon fatal errors.
4. **Resilient Auto-Resume:** Models save `checkpoint_latest.pth` each epoch and resume automatically upon server restarts.
5. **Frozen Preprocessing & Blind Test:** Internal test and external I-SPY1 sets remain sealed until the final model checkpoint freeze.
