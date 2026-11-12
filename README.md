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
  * **Development (Internal):** BreastDCEDL I-SPY2 ($n = 982$, 5-fold cross-validation from Krikun et al.; Fold 0: 785 train / 197 val).
  * **External Test:** I-SPY1 ($n = 171$, leakage-controlled zero-shot transfer).

---

## 🏆 Phase 1 Baselines: Official Benchmark Results (Fold 0, 197 Val Cases)

All 3 baseline architectures were trained on **I-SPY2 Fold 0** (785 train / 197 val) for 200 epochs on a single **NVIDIA L4 GPU**:

| Baseline Architecture | Mean Dice | Median Dice | Recall / Prec | Median HD95 | Status / Target | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Vanilla 3D U-Net** *(No Deep Supervision)* | `0.7129 ± 0.187` | `0.7668` | `0.804 / 0.694` | `9.43 mm` | ✅ Completed (GCP L4) | `240 MB` |
| **2. Standard 3D nnU-Net** *(With Deep Supervision)*| `0.7432 ± 0.171` | `0.7940` | `0.780 / 0.758` | `7.28 mm` | ✅ Completed (GCP L4) | `240 MB` |
| **3. ResEnc-UNet L** *(Residual Encoder - Baseline)* | **`0.7793 ± 0.164`** | **`0.8207`** | **`0.798 / 0.814`** | **`4.36 mm`** | ✅ Completed (GCP L4) | `1.08 GB` |
| *Elena Krikun et al. (EJR AI 2026, Table 2)* | `0.752 / 0.786` | `0.795 / 0.828` | `0.780 / 0.814` | `6.63 / 3.74 mm` | Reference Ground Truth | — |
| **4. SwinUNETR V1** *(Hierarchical 3D ViT - MONAI)* | `0.6916 ± 0.218` | `0.7454` | `0.742 / 0.738` | `11.20 mm` | ✅ Completed (GCP L4) | `720 MB` |
| **5. SwinUNETR V2** *(Stagewise ResConv + MergingV2)* | *Training...* | *Training...* | — | — | ⏳ In Progress (GCP L4) | `~730 MB` |
| **6. MedNeXt-L k3** *(Modern 3D CNN - MICCAI 2023)* | — | — | — | — | 🚀 Ready for School GPU | — |
| **7. U-Mamba Bot 3D** *(State Space Model - 2024)* | **`0.7481 ± 0.179`** | **`0.8040`** | `0.785 / 0.771` | **`6.32 mm`** | ✅ Completed (GCP L4) | `240 MB` |
| **8. PrimusV2** *(Pure 3D Transformer - 2024)* | — | — | — | — | 📋 Planned for Evaluation | — |

> 📁 Checkpoints, evaluation logs, and metric JSONs are stored locally in [`checkpoints/`](file:///d:/breastcancer_segmentation/checkpoints/) (see [`checkpoints/README.md`](file:///d:/breastcancer_segmentation/checkpoints/README.md) for full breakdown).

---

## 📚 Documentation & Engineering Guides

* [**University GPU Cluster & Lab Server Guidelines**](file:///d:/breastcancer_segmentation/UNIVERSITY_SERVER_GUIDE.md): Bản quy tắc vàng từ Thầy hướng dẫn khi SSH vào máy trường (cấm notebook, bắt buộc tmux, quản lý GPU/VRAM, phân vùng disk quota và checklist tiền kiểm tra).
* [**3D U-Net Best Practices & Lessons Learned**](file:///d:/breastcancer_segmentation/3D_UNET_BEST_PRACTICES.md): Đúc kết toàn bộ kinh nghiệm thực chiến, bài học chống sập RAM/OOM, bảo toàn Cross-Validation split, và quy chuẩn so sánh mô hình công bằng.
* [**Checkpoints & Benchmark Results**](file:///d:/breastcancer_segmentation/checkpoints/README.md): Bảng đối chiếu thực nghiệm trực diện 3 baseline (Vanilla 3D U-Net, Standard 3D nnU-Net, ResEnc-UNet L) so với Elena Krikun et al. (EJR AI 2026).
* [**Dataset Architecture Guide**](file:///d:/breastcancer_segmentation/DATASET_GUIDE.md): Cấu trúc 5 kênh động học $[I_0, I_e, I_l, \Delta I_e, \Delta I_l]$, phân tích dữ liệu I-SPY2 / I-SPY1, định dạng NIfTI và ánh xạ timepoint.
* [**GCP Cloud VM Operations Guide**](file:///d:/breastcancer_segmentation/GCP_GUIDE.md): Vận hành máy ảo GPU NVIDIA L4, quản trị Swap RAM, hạn chế worker I/O và hệ thống tự động tắt máy 2 tầng.
* [**Project Master Plan**](file:///d:/breastcancer_segmentation/README_PROJECT_PLAN.md): Đề cương nghiên cứu chi tiết, roadmap 2 giai đoạn (Segmentation First $\to$ Multi-Task Learning).


