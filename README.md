# 3D Breast DCE-MRI Tumor Segmentation with Auxiliary HER2 Classification

An empirical controlled study investigating whether patient-level clinical supervision (HER2 status) via a shared encoder improves 3D volumetric tumor segmentation in dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI).

---

## 🔬 Research Overview

* **Primary Question (RQ1):** Holding the backbone, data split, and training budget constant, does joint multi-task training with patient-level HER2 classification improve 3D tumor segmentation on Breast DCE-MRI over a controlled segmentation-only baseline?
* **Methodology:**
  * **Backbone:** 3D Residual Encoder nnU-Net (ResEnc).
  * **Input Representation:** 5 DCE-derived channels ($I_0, I_e, I_l, I_e - I_0, I_l - I_0$).
  * **Auxiliary Loss:** Patch-gated weighted binary cross-entropy ($L_{MTL} = L_{seg} + \lambda \cdot m_{cls} \cdot L_{cls}$), where classification loss is restricted to tumor-containing patches.
  * **Evaluation:** Paired bootstrap 95% CI and Wilcoxon signed-rank tests; raw predictions reported as primary metric.
* **Datasets:**
  * **Development (Internal):** BreastDCEDL I-SPY2 ($n = 982$, split 784 train / 99 val / 99 test).
  * **External Test:** I-SPY1 ($n = 171$, leakage-controlled zero-shot transfer).

---

## ⚙️ Quick Start

```bash
# 1. Activate virtual environment
# Windows: .\.venv\Scripts\Activate.ps1
# Linux:   source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
