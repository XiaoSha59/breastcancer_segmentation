# EXPERIMENT_TRACKER.md - SOTA 3D DCE-MRI Model Benchmarks & Execution Roadmap

> **Bản cập nhật tiến độ thực nghiệm**: 15/11/2026  
> **Mục tiêu**: Tái hiện và đánh giá 100% mã nguồn gốc các kiến trúc SOTA (CNN, Transformer, Mamba State Space) trên tập dữ liệu I-SPY2 DCE-MRI (Fold 0: 785 train / 197 val, 5 kênh động học, 200 epochs x 250 iter).

---

## ⛔ QUY TẮC BẤT DI BẤT DỊCH VỀ MẠNG TRÊN SERVER TRƯỜNG (`node1`)
- **SERVER TRƯỜNG KHÔNG KẾT NỐI GITHUB**: Máy chủ `node1` không có tài khoản/khóa kết nối GitHub, **TUYỆT ĐỐI CẤM dùng lệnh `git pull` hoặc `git clone` trên server trường**.
- **Cơ chế cập nhật mã nguồn**: Mọi cập nhật code từ máy cá nhân lên server trường được đưa qua đường SCP trung chuyển (hoặc user tự copy vào thư mục). Các file code mới nhất hiện **ĐÃ ĐƯỢC CẬP NHẬT TRỰC TIẾP VÀO SERVER TRƯỜNG**.
- **MÔI TRƯỜNG ẢO LÀ PYTHON VENV (`breast_env`) - KHÔNG DÙNG CONDA**: Trước khi chạy script, luôn kích hoạt bằng:
  ```bash
  source ~/breast_env/bin/activate
  ```
- **Khi chạy chỉ cần vào thư mục và gọi script trực tiếp trong `tmux`**, không gõ bất kỳ lệnh git hay conda nào.


---

## 1. BẢNG TIẾN ĐỘ TỔNG HỢP TOÀN BỘ CÁC MÔ HÌNH (MODEL PORTFOLIO)

| STT | Kiến trúc mô hình | Nguồn gốc mã nguồn | Trạng thái | Điểm Mean Dice | Điểm Median Dice | Vị trí Checkpoint & Kết quả (.pth, log) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **Standard 3D nnU-Net** | `nnU-Net v2` (MIC-DKFZ) | **HOÀN TẤT** | 0.7491 | 0.8012 | `checkpoints/standard_nnunet_fold0_checkpoint_best.pth` |
| 2 | **ResEnc-UNet L** | `nnU-Net v2 / dynamic-net` | **HOÀN TẤT** | 0.7513 | 0.8034 | `checkpoints/resenc_l_fold0_checkpoint_best.pth` |
| 3 | **Vanilla 3D U-Net** | `nnU-Net v2 (Plain Conv)` | **HOÀN TẤT** | 0.7180 | 0.7721 | `checkpoints/vanilla_3dunet_fold0_checkpoint_best.pth` |
| 4 | **U-Mamba Bot 3D** | `bowang-lab/U-Mamba` (MICCAI 2024) | **HOÀN TẤT** | 0.7481 | 0.8040 | `checkpoints/umamba_fold0_checkpoint_best.pth` |
| 5 | **SwinUNETR v1** | `MONAI (SwinTransformer)` | **HOÀN TẤT** | **0.7906** | **0.7950** | `checkpoints/swinunetr_fold0_checkpoint_best.pth` |
| 6 | **MedNeXt-L k3** | `facebookresearch/MedNeXt` | **HOÀN TẤT** | 0.7347 | 0.7924 | `results/.../MedNeXt_L_k3_200epochs/fold_0/`<br>`checkpoints/mednext_l_k3_fold0_checkpoint_best.pth` (414 MB) |
| 7 | **SegMamba 3D** | `ge-xing/SegMamba` (MICCAI 2024) | **ĐANG CHẠY (GCP L4)** | *(0.4162 tại Ep 4)* | -- | GCP VM `cancer-seg-gpu` (BFloat16, tmux `segmamba`) |
| 8 | **SwinUNETR v2** | `MONAI SwinUNETR (Standardized)` | **ĐANG CHẠY (node1)** | *(0.6140 tại Ep 39)* | -- | Server `node1` RTX 4090 (Ep 45+, tmux `swinunetr_v2`) |
| 9 | **PrimusV2-S** | `MIC-DKFZ` (Wald et al. 2024) | **SẴN SÀNG (node1)** | -- | -- | Đã vá 100%, chờ SwinUNETR v2 xong để chạy tiếp |

> *Ghi chú*: Toàn bộ 6 mô hình đã hoàn tất đều đã được kéo và lưu trữ an toàn cả file trọng số `.pth` (tốt nhất) lẫn file bảng điểm `paper_metrics_summary.txt` trên máy tính cá nhân (`D:\breastcancer_segmentation\checkpoints\`).

---

## 2. PHÂN BỔ THỰC THI HYBRID (SERVER TRƯỜNG & GCP)

Để tối ưu hóa phần cứng và rút ngắn thời gian làm đồ án từ 3 ngày xuống còn **1 ngày**, hai mô hình được phân bổ chạy song song:

```
+----------------------------------------------------------------------------------------------------------------+
| Model            | Môi trường chạy | Phần cứng & Driver           | Cơ chế đặc thù       | Tự động tắt máy (Shutdown)   |
+----------------------------------------------------------------------------------------------------------------+
| 1. SwinUNETR v2  | Server node1    | NVIDIA RTX 4090 24GB         | Native PyTorch CUDA  | KHÔNG TỰ TẮT (Cluster 24/7)  |
| 2. SegMamba 3D   | Google Cloud VM | NVIDIA L4 24GB (CUDA Toolkit)| Flash-Scan mamba_ssm | TỰ ĐỘNG TẮT KHI XONG (Tiết kiệm)|
| 3. PrimusV2-S    | Server node1    | NVIDIA RTX 4090 24GB         | ViT + 3D Axial RoPE  | Chạy sau khi SwinUNETR xong  |
+----------------------------------------------------------------------------------------------------------------+
```

- **Lợi ích chiến lược**:
  - **SwinUNETR v2** chạy trên server trường miễn phí, không vướng lỗi cuDNN nhờ cờ Native CUDA.
  - **SegMamba 3D** chạy trên GCP có sẵn `nvcc` và `mamba-ssm`, VRAM chỉ tốn ~10.8 GB, chạy xong tự động `poweroff` tiết kiệm 100% ngân sách.
  - **Cả 2 mô hình chạy song song**: Đến chiều Thứ Hai là có trọn vẹn kết quả của cả hai!

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
