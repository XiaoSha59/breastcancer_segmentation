# EXPERIMENT_TRACKER.md - SOTA 3D DCE-MRI Model Benchmarks & Execution Roadmap

> **Bản cập nhật tiến độ thực nghiệm**: 15/11/2026  
> **Mục tiêu**: Tái hiện và đánh giá 100% mã nguồn gốc các kiến trúc SOTA (CNN, Transformer, Mamba State Space) trên tập dữ liệu I-SPY2 DCE-MRI (Fold 0: 785 train / 197 val, 5 kênh động học, 200 epochs x 250 iter).

---

## ⛔ QUY TẮC BẤT DI BẤT DỊCH VỀ MẠNG TRÊN SERVER TRƯỜNG (`node1`)
- **SERVER TRƯỜNG KHÔNG KẾT NỐI GITHUB**: Máy chủ `node1` không có tài khoản/khóa kết nối GitHub, **TUYỆT ĐỐI CẤM dùng lệnh `git pull` hoặc `git clone` trên server trường**.
- **Cơ chế cập nhật mã nguồn**: Mọi cập nhật code từ máy cá nhân lên server trường được đưa qua đường SCP trung chuyển (hoặc user tự copy vào thư mục). Các file code mới nhất hiện **ĐÃ ĐƯỢC CẬP NHẬT TRỰC TIẾP VÀO SERVER TRƯỜNG**.
- **Khi chạy chỉ cần vào thư mục và gọi script trực tiếp trong `tmux`**, không gõ bất kỳ lệnh git nào.

---

## 1. BẢNG TIẾN ĐỘ TỔNG HỢP TOÀN BỘ CÁC MÔ HÌNH (MODEL PORTFOLIO)

| STT | Kiến trúc mô hình | Nguồn gốc mã nguồn | Trạng thái | Điểm Mean Dice | Điểm Median Dice | Vị trí Checkpoint & Kết quả (.pth, log) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **Standard 3D nnU-Net** | `nnU-Net v2` (MIC-DKFZ) | **HOÀN TẤT** | 0.7491 | 0.8012 | `checkpoints/standard_nnunet_fold0_checkpoint_best.pth` |
| 2 | **ResEnc-UNet L** | `nnU-Net v2 / dynamic-net` | **HOÀN TẤT** | 0.7513 | 0.8034 | `checkpoints/resenc_l_fold0_checkpoint_best.pth` |
| 3 | **Vanilla 3D U-Net** | `nnU-Net v2 (Plain Conv)` | **HOÀN TẤT** | 0.7180 | 0.7721 | `checkpoints/vanilla_3dunet_fold0_checkpoint_best.pth` |
| 4 | **U-Mamba Bot 3D** | `bowang-lab/U-Mamba` (MICCAI 2024) | **HOÀN TẤT** | 0.7481 | 0.8040 | `checkpoints/umamba_fold0_checkpoint_best.pth` |
| 5 | **SwinUNETR v1** | `MONAI (SwinTransformer)` | **HOÀN TẤT** | 0.6652 | 0.7104 | `checkpoints/swinunetr_fold0_checkpoint_best.pth` |
| 6 | **MedNeXt-L k3** | `facebookresearch/MedNeXt` | **HOÀN TẤT** | 0.7347 | 0.7924 | `results/.../MedNeXt_L_k3_200epochs/fold_0/`<br>`checkpoints/mednext_l_k3_fold0_checkpoint_best.pth` |
| 7 | **SegMamba 3D** | `ge-xing/SegMamba` (MICCAI 2024) | **CHƯA CHẠY** | -- | -- | Script: `scripts/run_segmamba_server.sh` |
| 8 | **SwinUNETR v2** | `MONAI SwinUNETR (Standardized)` | **CHƯA CHẠY** | -- | -- | Script: `scripts/run_swinunetr_v2_server.sh` |
| 9 | **PrimusV2-S** | `MIC-DKFZ` (Wald et al. 2024) | **CHƯA CHẠY** | -- | -- | Script: `scripts/run_primus_v2_server.sh` |

> *Ghi chú*: Toàn bộ 6 mô hình đã hoàn tất đều đã được kéo và lưu trữ an toàn cả file trọng số `.pth` (tốt nhất) lẫn file bảng điểm `paper_metrics_summary.txt` trên máy tính cá nhân (`D:\breastcancer_segmentation\`).

---

## 2. BA MÔ HÌNH CẦN HUẤN LUYỆN TRÊN SERVER TRƯỜNG (`node1`)

Cả 3 mô hình đều chạy trên card đồ họa **NVIDIA RTX 4090 24GB**:

```
+---------------------------------------------------------------------------------------------------+
| Model             | Loại kiến trúc              | VRAM ước tính | Tốc độ / Epoch | Tổng thời gian |
+---------------------------------------------------------------------------------------------------+
| 1. SegMamba 3D    | Pure 3D State Space Model   | ~10.8 - 11.2GB| ~5.0 - 5.5 min | ~19 tiếng      |
| 2. SwinUNETR v2   | Vision Transformer (MONAI)  | ~13.0 - 13.5GB| ~6.5 - 7.5 min | ~24 tiếng      |
| 3. PrimusV2-S     | EVA ViT + 3D Axial RoPE     | ~11.5 - 12.2GB| ~6.0 - 7.0 min | ~22 tiếng      |
+---------------------------------------------------------------------------------------------------+
```

- **Ràng buộc phần cứng**: Vì mỗi mô hình chiếm từ 11GB đến 13.5GB VRAM, card 24GB **không thể chạy song song 2 mô hình cùng lúc**. Bắt buộc phải chạy **tuần tự (sequential)**.
- **Tổng thời gian chạy 3 mô hình**: Khoảng 65 tiếng (~2.7 ngày).

---

## 3. LỊCH TRÌNH CĂN GIỜ TỐI ƯU (KHỚP VỚI GIỜ HÀNH CHÍNH & GIỜ NGHỈ)

- **Khung giờ rảnh kiểm tra**: 07:30 - 08:30 (sáng trước giờ làm việc) và 17:30 - 22:30 (sau giờ hành chính).
- **Khung giờ không thể kiểm tra**: 08:30 - 17:30 (giờ hành chính) và 23:00 - 07:00 (đêm ngủ).

### Lộ trình chi tiết:

```
[Chủ Nhật 13:30]
       │
       ▼
 1. BẮT ĐẦU CHẠY SEGMAMBA 3D (~19 tiếng)
       │ (Tự chạy nền qua đêm trong tmux)
       ▼
[Thứ Hai 08:00 - 08:30] (TRƯỚC GIỜ HÀNH CHÍNH)
       │ -> SegMamba HOÀN TẤT (200 ep + 197 ca val)
       │ -> Bạn mở máy kiểm tra bảng điểm, bắn sang GCP kéo về ổ D:
       │
       ▼ (Trong ngày Thứ Hai: GPU nghỉ hoặc chuyển sang model 2)
[Thứ Hai 18:00] (SAU GIỜ HÀNH CHÍNH)
       │
       ▼
 2. BẮT ĐẦU CHẠY SWINUNETR V2 (~24 tiếng)
       │ (Tự chạy nền qua đêm trong tmux)
       ▼
[Thứ Ba 18:00] (SAU GIỜ HÀNH CHÍNH)
       │ -> SwinUNETR v2 HOÀN TẤT
       │ -> Bạn đi làm về mở máy kiểm tra kết quả & kéo về ổ D:
       │
       ▼
[Thứ Ba 19:00]
       │
       ▼
 3. BẮT ĐẦU CHẠY PRIMUSV2-S (~22 tiếng)
       │ (Tự chạy nền qua đêm trong tmux)
       ▼
[Thứ Tư 17:00 - 18:00] (SAU GIỜ HÀNH CHÍNH)
       │ -> PrimusV2-S HOÀN TẤT
       │ -> Toàn bộ danh mục 9 mô hình SOTA được tái hiện trọn vẹn 100%!
```

---

## 4. HƯỚNG DẪN KÍCH HOẠT LỆNH TRÊN SERVER TRƯỜNG (`node1`)

### Lựa chọn A: Kích hoạt từng mô hình thủ công

#### Bước 1: Chạy SegMamba 3D
```bash
cd ~/breastcancer_segmentation
tmux new -s segmamba
bash scripts/run_segmamba_server.sh
```
*(Bấm `Ctrl + B`, thả tay ra ấn `D` để thoát màn hình tmux, tiến trình vẫn chạy ngầm 24/7).*

#### Bước 2: Chạy SwinUNETR v2 (Sau khi SegMamba xong)
```bash
cd ~/breastcancer_segmentation
tmux new -s swinunetr_v2
bash scripts/run_swinunetr_v2_server.sh
```

#### Bước 3: Chạy PrimusV2-S (Sau khi SwinUNETR v2 xong)
```bash
cd ~/breastcancer_segmentation
tmux new -s primus_v2
bash scripts/run_primus_v2_server.sh
```

---

### Lựa chọn B: Chạy Hàng Đợi Nối Tiếp Tự Động (Auto-Queue)
Chỉ cần bấm lệnh 1 lần duy nhất trong `tmux`, 3 mô hình sẽ tự động nối đuôi nhau chạy:

```bash
cd ~/breastcancer_segmentation
tmux new -s sota_queue
bash scripts/run_sota_queue.sh
```

---

## 5. NGUYÊN TẮC AN TOÀN BẤT DI BẤT DỊCH TRÊN SERVER TRƯỜNG
1. **Không tự ý tắt máy (Zero auto-shutdown)**: Server trường là tài nguyên chung, cấm mọi lệnh tắt máy.
2. **Không xóa bừa bãi**: Tuyệt đối không xóa dữ liệu ngoài thư mục `~/breastcancer_segmentation/`.
3. **Giới hạn luồng CPU**: Luôn giữ `OMP_NUM_THREADS=4` và `torch.set_num_threads(4)`.
4. **Bảo toàn dữ liệu chia tập**: Cố định 100% `splits_final.json` (785 train / 197 val), tuyệt đối không sinh lại split.
5. **Dung lượng đĩa gọn gàng**: Mỗi mô hình chỉ lưu 2 file trọng số (`checkpoint_best.pth` và `checkpoint_latest.pth`), tổng dung lượng mỗi mô hình dưới 2 GB.
