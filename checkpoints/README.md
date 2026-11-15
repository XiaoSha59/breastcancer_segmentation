# 📦 Model Checkpoints & Official Reproduction Results

Thư mục lưu trữ toàn bộ trọng số mô hình tốt nhất (`checkpoint_best.pth`), biểu đồ huấn luyện và kết quả nghiệm thu khoa học chính thức trên **I-SPY2 Fold 0 (197 ca kiểm định)**.

---

## 🏆 BẢNG TỔNG HỢP TOÀN BỘ DANH MỤC 9 MÔ HÌNH SOTA (FOLD 0)

| STT | Kiến trúc mô hình | Phân loại kiến trúc | Trạng thái | Mean Dice | Median Dice | Median HD95 | File Checkpoint & Logs |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **ResEnc-UNet L** | Pure CNN (Residual Blocks) | **HOÀN TẤT** | **0.7793** | **0.8207** | **4.36 mm** | `resenc_l_fold0_checkpoint_best.pth` (1.08 GB) |
| 2 | **Standard 3D nnU-Net** | Pure CNN (Plain Conv + DS) | **HOÀN TẤT** | 0.7432 | 0.7940 | 7.28 mm | `standard_nnunet_fold0_checkpoint_best.pth` (240 MB) |
| 3 | **MedNeXt-L k3** | Pure CNN (ConvNeXt Inverted) | **HOÀN TẤT** | 0.7347 | 0.7924 | 6.40 mm | `mednext_l_k3_fold0_checkpoint_best.pth` (414 MB) |
| 4 | **Vanilla 3D U-Net** | Pure CNN (Plain Conv, No DS) | **HOÀN TẤT** | 0.7129 | 0.7668 | 9.43 mm | `vanilla_3dunet_fold0_checkpoint_best.pth` (240 MB) |
| 5 | **U-Mamba Bot 3D** | Hybrid CNN + Mamba SSM | **HOÀN TẤT** | 0.7481 | 0.8040 | 6.32 mm | `umamba_fold0_checkpoint_best.pth` (337 MB) |
| 6 | **SwinUNETR v1** | Shifted Window ViT | **HOÀN TẤT** | 0.6652 | 0.7104 | 9.85 mm | `swinunetr_fold0_checkpoint_best.pth` (754 MB) |
| 7 | **SwinUNETR v2** | ResConv + Swin ViT v2 | **ĐANG CHẠY (node1)** | *(Đạt 0.6140 tại Ep 39)* | -- | -- | `swinunetr_v2_fold0_checkpoint_best.pth` (880 MB) |
| 8 | **SegMamba 3D** | Tri-orientated 3D Mamba | **ĐANG CHẠY (GCP L4)** | *(Đạt 0.4162 tại Ep 4)* | -- | -- | GCP VM `cancer-seg-gpu` (BFloat16) |
| 9 | **PrimusV2-S** | Pure ViT + 3D Axial RoPE | **SẴN SÀNG (node1)** | -- | -- | -- | Script: `scripts/run_primus_v2_server.sh` |

---

## 🔍 ĐỐI CHIẾU BỨC TRANH TOÀN CẢNH SOTA NĂM 2026: CÓ THIẾU MÔ HÌNH NÀO KHÔNG?

Trong các nghiên cứu y khoa hàng đầu (MICCAI 2024, 2025, 2026, IEEE TMI, Nature Methods), toàn bộ các kiến trúc phân đoạn ảnh y tế 3D được phân định thành **3 trường phái công nghệ trụ cột**:

### 1. Trường phái Pure CNN (Mạng Tích chập Thuần túy)
- **Đại diện kinh điển**: `Standard nnU-Net` (Isensee et al., Nature Methods 2021) & `Vanilla 3D U-Net`. -> **ĐÃ CÓ (Hoàn tất)**
- **Đại diện SOTA hiện đại**: `ResEnc-UNet L` (Isensee et al., 2024) & `MedNeXt` (Roy et al., MICCAI 2023/2024 - ConvNeXt Inverted Bottleneck). -> **ĐÃ CÓ (Hoàn tất cả 2)**

### 2. Trường phái Vision Transformers (Cơ chế Chú ý ViT)
- **Đại diện Shifted Window**: `SwinUNETR v1` (Hatamizadeh et al., MONAI) & `SwinUNETR v2` (kết hợp Residual Conv Blocks, 2024). -> **ĐÃ CÓ (v1 xong, v2 đang chạy)**
- **Đại diện Pure ViT 3D tiên tiến nhất**: `PrimusV2` (Wald, Roy, Isensee, Maier-Hein - DKFZ, MICCAI 2024 / arXiv 2025) - sử dụng 3D Axial Rotary Position Embedding (RoPE) và Iterative Residual Tokenizer. -> **ĐÃ CÓ (Sẵn sàng chạy tiếp trên node1)**

### 3. Trường phái State Space Models / Mamba (Quét Chọn lọc Tuyến tính 2024 - 2026)
- **Dạng Hybrid CNN-Mamba**: `U-Mamba` (Ma et al., MICCAI 2024) - cấy Mamba vào nút cổ chai Bottleneck của U-Net. -> **ĐÃ CÓ (Hoàn tất, Dice 0.7481)**
- **Dạng Pure 3D Mamba**: `SegMamba` (Wang et al., MICCAI 2024) - quét không gian 3 chiều độc lập (Tri-orientated 3D Mamba). -> **ĐÃ CÓ (Đang chạy trên GCP)**

### Kết luận đánh giá SOTA 2026:
* Danh mục 9 mô hình trong dự án đã **bao phủ 100% tất cả các đại diện SOTA xuất sắc nhất** đại diện cho cả 3 trường phái công nghệ tính đến năm 2026.
* Các mô hình khác trong y văn (như `nnMamba` thực chất có kiến trúc tương đồng với `U-Mamba`, hoặc `MedSAM` là mô hình Foundation Model cần prompt thủ công không tự động phân đoạn DCE-MRI 5 pha) đều không tạo ra giá trị so sánh cốt lõi bằng 9 kiến trúc chính này.

---

## 📁 Danh Mục Các Artifacts Đã Lưu Tại Máy Local (`D:\breastcancer_segmentation\`):

1. **ResEnc-UNet L (Flagship Baseline)**:
   - `checkpoints/resenc_l_fold0_checkpoint_best.pth` (1.08 GB)
   - `checkpoints/resenc_l_fold0_evaluation_summary.txt` (Điểm Dice 0.7793, IoU 0.6617, HD95 4.36 mm)
2. **Standard 3D nnU-Net (Ablation Baseline 1)**:
   - `checkpoints/standard_nnunet_fold0_checkpoint_best.pth` (240 MB)
   - `checkpoints/standard_nnunet_fold0_evaluation_summary.txt` (Điểm Dice 0.7432, HD95 7.28 mm)
3. **Vanilla 3D U-Net (Ablation Baseline 2)**:
   - `checkpoints/vanilla_3dunet_fold0_checkpoint_best.pth` (240 MB)
   - `checkpoints/vanilla_3dunet_fold0_evaluation_summary.txt` (Điểm Dice 0.7129, HD95 9.43 mm)
4. **MedNeXt-L k3 (Modern ConvNeXt SOTA)**:
   - `checkpoints/mednext_l_k3_fold0_checkpoint_best.pth` (414 MB)
   - `results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0/paper_metrics_summary.txt` (Dice 0.7347, Median 0.7924)
5. **U-Mamba Bot 3D (Hybrid SSM SOTA)**:
   - `checkpoints/umamba_fold0_checkpoint_best.pth` (337 MB)
   - `checkpoints/umamba_fold0_evaluation.txt` (Dice 0.7481, Median 0.8040)
6. **SwinUNETR v1 (Transformer Baseline)**:
   - `checkpoints/swinunetr_fold0_checkpoint_best.pth` (754 MB)
   - `checkpoints/swinunetr_fold0_training_summary.json` (Trọn vẹn 200 epochs)
