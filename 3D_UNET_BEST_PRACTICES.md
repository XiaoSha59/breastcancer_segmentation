# 🏆 3D U-Net & Medical Segmentation: Best Practices & Kinh Nghiệm Thực Chiến

Tài liệu đúc kết toàn bộ kinh nghiệm thực tế, bài học xương máu và các quy chuẩn khoa học tốt nhất khi huấn luyện các mô hình phân đoạn ảnh y tế 3D (3D DCE-MRI Tumor Segmentation) với **nnU-Net v2**, **Residual Encoder UNet (ResEnc-UNet L)** và **Standard 3D nnU-Net** trên tập dữ liệu I-SPY2 / Breast MRI.

---

## 0. ⛔ NGUYÊN TẮC BẤT KHẢ XÂM PHẠM: BẮT BUỘC DÙNG MÃ NGUỒN GỐC CÔNG BỐ CỦA TÁC GIẢ – TUYỆT ĐỐI KHÔNG TỰ CHẾ

Trong nghiên cứu khoa học và thực nghiệm so sánh mô hình (Benchmark / Baseline Comparison):
1. **Tuyệt đối KHÔNG tự ý viết lại (re-implement), đơn giản hóa hay tự chế kiến trúc mô hình**:
   - Mọi mô hình so sánh (MedNeXt, SwinUNETR, Primus, U-Mamba, SegMamba,...) **BẮT BUỘC** phải được lấy trực tiếp 100% nguyên bản từ repository chính thức của nhóm tác giả đã công bố tại các hội nghị/tạp chí uy tín (MICCAI, CVPR, ICCV, Nature, IEEE) hoặc từ các thư viện chuẩn quốc tế đã kiểm định (MONAI, nnU-Net, dynamic-network-architectures).
2. **Không tự ý thay đổi cơ chế toán tử**:
   - Không được tự ý thay đổi cách tokenize, cách quét chiều không gian, các khối trích xuất đặc trưng cục bộ (như GSC trong SegMamba, Inverted Bottleneck trong MedNeXt, hay Shifted Window trong SwinUNETR).
3. **Chỉ được phép điều chỉnh đúng tham số kích thước đầu vào / đầu ra của bài toán**:
   - `in_channels` (ví dụ: 5 pha DCE-MRI thay vì ảnh não 4 kênh).
   - `num_classes` (ví dụ: 2 lớp Background, Tumor thay vì đa lớp).
   - `patch_size` phù hợp bộ nhớ card GPU theo đúng chuẩn kế hoạch của nnU-Net.
4. **Hậu quả nếu tự chế kiến trúc**:
   - Sai lệch bản chất mô hình, làm vô hiệu hóa giá trị so sánh khoa học của bài báo/luận văn.
   - Nguy cơ bùng nổ bộ nhớ Autograd, tràn VRAM và làm sập máy chủ phòng lab.

---

## 1. Nguyên Tắc So Sánh Mô Hình Công Bằng (Fair Baseline Comparison)

Khi thực hiện nghiên cứu khoa học hoặc viết báo cáo học thuật, việc so sánh giữa các kiến trúc (Ablation Study / Baseline Comparison) đòi hỏi tính kiểm soát biến số nghiêm ngặt:

| Tiêu chí | Quy chuẩn bắt buộc | Giải thích lý do |
| :--- | :--- | :--- |
| **Số lượng Epochs** | Cố định **200 epochs** (hoặc 1000) cho **tất cả mô hình** | nnU-Net mặc định chạy 1.000 epochs. Cần ép cờ `-tr nnUNetTrainer_200epochs` cho cả ResEnc và Standard nnU-Net để không bị thiên lệch do thời lượng học khác nhau. |
| **Số Iterations / Epoch** | Giữ nguyên **250 iterations/epoch** | Không tùy tiện giảm xuống 100 iter/epoch vì sẽ làm giảm số lần lấy mẫu patch khối u, gây rớt điểm Dice khi quét toàn thể tích. |
| **Learning Rate Schedule** | Poly-decay: $\eta = \eta_0 \times (1 - \text{epoch}/\text{max\_epoch})^{0.9}$ | Đảm bảo bước nhảy gradient thu nhỏ dần một cách tương đương ở mọi mô hình. |
| **Biểu diễn Kênh (Input)** | Đồng nhất **5 kênh động học** $[I_0, I_e, I_l, \Delta I_e, \Delta I_l]$ | Tránh so sánh khập khiễng giữa mô hình 3 kênh và mô hình 5 kênh. |

---

## 2. Bảo Toàn Tính Toàn Vẹn Của Tập Dữ Liệu (Cross-Validation Integrity)

### ⚠️ Rủi Ro Chí Mạng: Lệch Tập Split (`splits_final.json`)
* **Bản chất:** Muốn chứng minh *ResEnc-UNet L* vượt trội hơn *Standard 3D nnU-Net*, bạn phải chứng minh trên **cùng một tập bệnh nhân kiểm thử**.
* **Cạm bẫy trong nnU-Net:** Khi bạn chạy lệnh lập kế hoạch (`nnUNetv2_plan_experiment`) hoặc tiền xử lý cho một kiến trúc mới, nnU-Net có thể tự động sinh ngẫu nhiên một file `splits_final.json` mới nếu file cũ bị xóa hoặc ghi đè.
* **Hậu quả:** 197 bệnh nhân ở Fold 0 của mô hình A sẽ khác hoàn toàn 197 bệnh nhân ở Fold 0 của mô hình B $\to$ **Mọi kết luận so sánh đều bị vô hiệu hóa về mặt khoa học!**

### ✅ Best Practice Thực Hiện:
1. Tạo và kiểm tra file `splits_final.json` chuẩn một lần duy nhất tại thư mục `$nnUNet_preprocessed/<DatasetName>/`.
2. Luôn sao lưu một bản dự phòng tại thư mục dự án (ví dụ: `data/processed/splits_final.json`).
3. Trong mọi script khởi chạy mô hình mới, luôn có lệnh kiểm tra và khôi phục split:
   ```bash
   cp /home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json \
      "$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"
   ```

---

## 3. Tái Sử Dụng Dữ Liệu Tiền Xử Lý (Preprocessed Data Reuse)

* **Vấn đề:** Dữ liệu ảnh 3D MRI sau khi nội suy resampled thành file `.npz` và `.npy` rất nặng (khoảng 30 – 50 GB). Nếu mỗi mô hình (Standard nnU-Net, ResEnc-UNet L, Plain U-Net) lại chạy lại `nnUNetv2_preprocess` từ đầu, bạn sẽ:
  * Tốn thêm **2 – 3 tiếng** tính toán CPU vô ích.
  * Tốn thêm **30 – 50 GB dung lượng SSD** cho mỗi cấu hình (dễ dẫn đến đầy đĩa `Disk Full`).
* **Kinh nghiệm:**
  * Cả Standard 3D nnU-Net (`nnUNetPlans`) và ResEnc-UNet L (`nnUNetResEncUNetLPlans`) trên cùng bài toán 3D fullres đều dùng chung độ phân giải và spacing mục tiêu.
  * Chỉ cần sinh file mô tả kế hoạch mạng (`nnUNetPlans.json`) mà **không cần chạy lại bước `nnUNetv2_preprocess`** nếu thư mục `nnUNetPlans_3d_fullres` đã tồn tại.

### 🌟 Nghiên Cứu Bóc Tách 3 Pha (3-Phase Ablation) Cho Họ nnU-Net
* **Thách thức:** Dữ liệu Dataset 501 có 5 kênh và chiếm 64 GB bộ nhớ đệm preprocessed. Nếu tạo thêm một dataset 3 kênh riêng biệt, máy chủ trường sẽ cạn sạch dung lượng SSD (vượt ngưỡng 82 GB trống).
* **Giải pháp chuẩn MIC-DKFZ:** Sử dụng Trainer kế thừa `nnUNetTrainer_3phases_200epochs`:
  * Khởi tạo mạng với `num_input_channels = 3`.
  * Cắt lát động tensor đầu vào `batch['data'] = batch['data'][:, :3]` tại bước nạp dữ liệu.
  * **Hiệu quả:** Tái sử dụng 100% dữ liệu 64 GB đã xử lý, tiết kiệm 0 MB ổ cứng phát sinh, đảm bảo so sánh 1:1 công bằng tuyệt đối giữa 5 pha và 3 pha.

---

## 4. Quản Lý Bộ Nhớ & Tránh Lỗi Sập RAM (OOM Crash Prevention)

### 💥 Bài Học Xương Máu: Sập OOM ở Epoch 5
Trong ảnh y tế 3D, các patch thể tích (ví dụ $128 \times 128 \times 128$ hoặc $256 \times 256 \times 80$) có kích thước bộ nhớ lớn gấp hàng chục lần ảnh 2D thông thường.

1. **Khống Chế Số Lượng Worker Tăng Cường Dữ Liệu (`nnUNet_n_proc_DA`):**
   * Mặc định nnU-Net sử dụng **8 workers** để prefetch và xoay lật ảnh 3D trong RAM. Trên máy có 32 GB RAM, 8 workers đồng thời sẽ đẩy mức tiêu thụ bộ nhớ ảo vượt quá 42 GB $\to$ Linux OOM Killer sẽ thẳng tay hạ sát tiến trình (`Killed process nnUNetv2_train`).
   * **Quy tắc bắt buộc:** Luôn khai báo biến môi trường:
     ```bash
     export nnUNet_n_proc_DA=4
     export OMP_NUM_THREADS=1
     ```
   * *Kết quả:* RAM hệ thống duy trì cực kỳ phẳng và ổn định ở mức ~10 GB / 31 GB.

2. **Cấu Hình Swap File Tối Thiểu 32 GB:**
   * Luôn tạo vùng đệm ảo (Swap space) tối thiểu ngang bằng hoặc gấp đôi RAM vật lý:
     ```bash
     sudo fallocate -l 32G /swapfile
     sudo chmod 600 /swapfile
     sudo mkswap /swapfile
     sudo swapon /swapfile
     ```

---

## 5. Đánh Giá Hội Tụ: Phân Biệt "Pseudo Dice" vs "Validation Cohort"

Khi theo dõi log huấn luyện, rất nhiều người nhầm lẫn giữa điểm Dice in trong lúc train và điểm Dice thực tế:

| Khái niệm | Pseudo Dice (Trong lúc Train) | Full Validation Dice (Nghiệm thu sau Train) |
| :--- | :--- | :--- |
| **Phương thức lấy mẫu** | Cắt nhanh **50 patch ngẫu nhiên** mỗi epoch trên các vùng u | Quét **trượt 3D (Sliding Window, 50% overlap)** trên **toàn bộ thể tích gốc** của 197 bệnh nhân |
| **Ý nghĩa kỹ thuật** | Chỉ là hàm đại diện (proxy metric) để lọc nhiễu và lưu `checkpoint_best.pth` qua đường EMA | Là điểm số chính xác 100% để báo cáo khoa học |
| **Các chỉ số kèm theo** | Chỉ có 1 con số Dice ước lượng | Có thêm **HD95 (95% Hausdorff Distance)** và **ASSD (Average Symmetric Surface Distance)** tính bằng milimét |

### 4 Dấu Hiệu Chứng Minh Mô Hình Đang Hội Tụ Đúng Chuẩn:
1. `val_loss` và `train_loss` liên tục đào sâu và **chênh lệch (gap) $\le 0.03$** (chứng minh không bị Overfitting).
2. Đường **`EMA Pseudo Dice`** tăng đơn điệu và phẳng dần (plateau) về cuối đợt train.
3. Tốc độ biến thiên loss chậm dần ở 20 epoch cuối (diminishing returns).
4. `learning_rate` giảm sâu (từ $0.01$ xuống $< 0.0001$), giúp mạng hội tụ mịn vào đáy cực tiểu toàn cục.

---

## 6. Tự Động Hóa & Tối Ưu Chi Phí Đám Mây (Cloud Cost Engineering)

Huấn luyện ảnh 3D kéo dài hàng chục giờ đòi hỏi quy trình tự động hóa khép kín để không lãng phí ngân sách thuê GPU:

1. **Hệ Thống Tự Động Tắt Máy 2 Lớp (Fail-Safe Shutdown):**
   * *Lớp 1 (Script Trap):* Cài `trap cleanup EXIT ERR INT TERM` trong bash script để bắt mọi sự kiện kết thúc (kể cả lỗi crash). Chờ 120 giây xả buffer rồi thực thi `sudo /sbin/shutdown -h now`.
   * *Lớp 2 (Daemon Watchdog):* Một tiến trình nền độc lập (`safety_guard.py`) kiểm tra trạng thái PID huấn luyện mỗi 60 giây. Nếu tiến trình chính biến mất quá 5 phút, tự động kích hoạt `sudo shutdown -h now`.
2. **Nối Tiếp Khâu Đánh Giá (Seamless Evaluation Chaining):**
   * Không nên để máy tắt ngay sau Epoch cuối cùng rồi sáng hôm sau bật máy lên chạy lại lệnh `--val`.
   * Sử dụng pipeline điều phối tự động (`compute_paper_metrics.py`) để ngay khi kết thúc Epoch 199, máy ảo lập tức chạy tiếp 30 phút Validation Inference trên 197 ca, tính đủ 4 chỉ số Paper (Mean/Median Dice, HD95, ASSD) rồi mới tắt nguồn.
   * *Lợi ích:* Tiết kiệm thời gian khởi động, tận dụng CUDA cache và sáng hôm sau có sẵn kết quả nghiệm thu hoàn chỉnh!

---

## 7. Cạm Bẫy Kỹ Thuật Khi Tạo Custom Trainer Trong nnU-Net v2

### ⚠️ Bài Học Xương Máu: Lỗi `KeyError: 'args'` Do Cơ Chế Reflection
Khi kế thừa class `nnUNetTrainer` để tạo trainer tùy biến (ví dụ đổi số epoch, đổi hàm loss, hoặc tắt Deep Supervision):
* **Sai lầm phổ biến:** Sử dụng cú pháp gom tham số thông thường trong Python:
  ```python
  # ❌ SAI LẦM: GÂY SẬP NGAY TẠI DÒNG ĐẦU TIÊN
  class nnUNetTrainer_Custom(nnUNetTrainer):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
  ```
* **Nguyên nhân bên dưới:** Bên trong `nnUNetTrainer.__init__`, tác giả nnU-Net dùng cơ chế nội soi (reflection):
  `inspect.signature(self.__class__.__init__).parameters` để lấy tên các tham số rồi truy xuất trực tiếp qua từ điển `locals()[k]`.
  Khi ta dùng `*args, **kwargs`, bộ duyệt tìm thấy biến có tên là `'args'` và tra cứu `locals()['args']` $\to$ Gây lỗi văng chương trình: `KeyError: 'args'`!
* **✅ Quy chuẩn chuẩn hóa (Best Practice):** Bắt buộc phải khai báo danh sách tham số **tường minh 100%**:
  ```python
  class nnUNetTrainer_Vanilla3DUNet_200epochs(nnUNetTrainerNoDeepSupervision):
      def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                   device: torch.device = torch.device('cuda')):
          super().__init__(plans, configuration, fold, dataset_json, device)
          self.enable_deep_supervision = False
          self.num_epochs = 200
  ```

---

## 8. Đúc Kết Bài Học Khoa Học Từ Chuỗi 3 Baseline (Ablation Insights)

Kết quả thực nghiệm trên 197 ca kiểm định Fold 0 (I-SPY2) tạo nên bức tranh đối chứng hoàn hảo:

$$\text{Vanilla 3D U-Net (0.7129)} \xrightarrow{+\text{Deep Supervision}} \text{Standard nnU-Net (0.7432)} \xrightarrow{+\text{Residual Encoder}} \text{ResEnc-UNet L (0.7793)}$$

| Chỉ số Paper | Vanilla 3D U-Net<br>*(No Deep Supervision)* | Standard 3D nnU-Net<br>*(With Deep Supervision)* | ResEnc-UNet L<br>*(Residual Encoder)* | Krikun et al.<br>*(EJR AI 2026 Table 2)* |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Dice** | `0.7129 ± 0.1867` | `0.7432 ± 0.1713` | **`0.7793 ± 0.1644`** | `0.752` vs `0.786` |
| **Median Dice** | `0.7668` | `0.7940` | **`0.8207`** | `0.795` vs `0.828` |
| **Recall / Precision** | `0.8041` / `0.6940` | `0.7797` / `0.7582` | **`0.7980` / `0.8140`** | `0.780` / `0.770` |
| **Median HD95** | `9.43 mm` | `7.28 mm` | **`4.36 mm`** | `6.63 mm` vs `3.74 mm` |
| **Median ASSD** | `1.99 mm` | `1.50 mm` | **`0.93 mm`** | `1.43 mm` vs `0.95 mm` |

### 2 Phát Hiện Học Thuật Đắt Giá Cho Luận Văn / Bài Báo:
1. **Hiện tượng Over-segmentation khi thiếu Deep Supervision:**
   * Ở Vanilla 3D U-Net, Recall đạt rất cao (`0.8041`), nhưng Precision tụt sâu xuống `0.6940` và Median HD95 phình to lên `9.43 mm`.
   * *Bản chất:* Thiếu Deep Supervision dẫn đến suy giảm gradient ở các tầng nông; mạng không thể định hình sắc nét đường viền ngoài mà phân vùng tràn lan ra mô mỡ/mô lành xung quanh. Bật Deep Supervision giúp kéo sai số HD95 từ `9.43 mm` xuống `7.28 mm` (-22.8%).
2. **Sức mạnh định hình đường viền của Residual Blocks:**
   * Khi nâng cấp lên ResEnc-UNet L, HD95 giảm một bước ngoặt khổng lồ từ `7.28 mm` xuống **`4.36 mm`** (-40.1%), và khoảng cách bề mặt ASSD lần đầu tiên lọt xuống dưới 1 mm (**`0.93 mm`**).
   * Điều này chứng minh Residual Connections cho phép mạng tăng gấp rưỡi dung lượng (lên 31M tham số) mà không bị bão hòa gradient, giúp phân định ranh giới vi mô của khối u ngấm thuốc cực kỳ chuẩn xác.

---

## 9. Mở Rộng Kiến Trúc SOTA Hiện Đại (MedNeXt-L k3, SwinUNETR, U-Mamba)

Để mở rộng nghiên cứu từ baseline cổ điển sang các trường phái kiến trúc tối tân nhất (2023 - 2024), quy chuẩn thiết kế và vận hành được mở rộng như sau:

### 9.1. Tại sao chọn MedNeXt-L k3 thay vì k5 cho Breast DCE-MRI?
1. **Chống hiện tượng mờ nhẵn bờ vi thể (Oversmoothing):** 
   * Khối u ác tính ở vú có ranh giới vi mô phức tạp (bờ tua gai spiculated, bờ đa cung vi thể, đốm ngấm thuốc động học NME). Kernel $5 \times 5 \times 5$ có xu hướng làm mờ (smoothing) các ranh giới này, trong khi kernel $3 \times 3 \times 3$ bảo toàn độ tương phản cường độ cục bộ sắc nét.
2. **Tương thích với Spacing dị hướng (Anisotropic Spacing):** 
   * Ảnh DCE-MRI có chiều dày lát cắt trục $Z$ thưa hơn mặt phẳng $X, Y$. Ở các tầng sâu (patch $128 \times 128 \times 64$), trục $Z$ chỉ còn 4 – 8 voxels. Kernel $5 \times 5 \times 5$ sẽ gây biến dạng biên (padding artifacts) nghiêm trọng, trong khi $3 \times 3 \times 3$ trượt rất êm và tự nhiên.
3. **Cơ chế Inverted Bottleneck & Residual Scaling:**
   * MedNeXt áp dụng mở rộng kênh $1 \times 1 \times 1 \to 3 \times 3 \times 3 \text{ Depthwise Conv} \to \text{GELU} \to 1 \times 1 \times 1 \text{ Pointwise Conv}$, kết hợp Deep Supervision đa tỉ lệ, mang lại năng lực trích xuất đặc trưng vượt trội so với ResEnc-UNet L truyền thống.

### 9.2. Quy Chuẩn Đọc Dữ Liệu Kép (.b2nd Blosc2 & .npy)
* Thư mục tiền xử lý chuẩn `nnUNetPlans_3d_fullres` của nnU-Net v2 hiện đại lưu mảng ảnh nén Blosc2 (`.b2nd`) gồm 2 file tách biệt: `{cid}.b2nd` (5 channels ảnh) và `{cid}_seg.b2nd` (1 channel mặt nạ u).
* Mọi bộ nạp `FastNpyBatchSampler` bắt buộc phải tự động nhận diện cả 2 định dạng `.b2nd` (dùng `blosc2.open`) và `.npy` (dùng `np.load(..., mmap_mode='r')`) để đảm bảo tính di động 100% trên mọi máy chủ.

### 9.3. Vận Hành An Toàn Trên Máy Chủ Trường / Cụm HPC (Cluster Mode)
* **Quy tắc bất khả xâm phạm:** Tuyệt đối KHÔNG giải phóng ổ đĩa, không chạy các lệnh `rm`, `clean`, `purge`, `apt clean` và không làm ảnh hưởng bất kỳ file/dữ liệu hiện có của các thành viên khác trên server. Toàn bộ tài nguyên của project được gói gọn trong thư mục cá nhân.
* **Quy tắc tuyệt đối:** Không tự động kích hoạt `sudo shutdown` / `sudo poweroff` trên máy chủ dùng chung (toàn bộ code runner đã được gỡ sạch các lệnh này).
* **Quản lý tiến trình ngầm:** Bắt buộc sử dụng `tmux` (`tmux new -s train_job`) để tránh tiến trình bị ngắt tín hiệu `SIGHUP` khi đóng trình duyệt hoặc ngắt kết nối SSH.

---

### 9.4. ⚠️ Bài Học Xương Máu Về Đánh Giá (Evaluation I/O: SimpleITK vs Blosc2 Buffer Error)

Trong giai đoạn kiểm thử mô hình (Inference trên 197 ca Validation):

#### 1. Hiện Tượng & Cạm Bẫy Kỹ Thuật (Sự Cố Ca Ngoại Cỡ `ISPY2-987995`):
* Trong tập dữ liệu, một số bệnh nhân có thể tích ma trận 3D ngoại cỡ (ví dụ ca `ISPY2-987995` có **160 lát cắt Z** với ma trận $512 \times 512$).
* Khi đọc từ file nén tốc độ cao Blosc2 (`.b2nd`), lệnh đọc nguyên khối 3D theo kênh `val_arr[c] = img_h[c]` yêu cầu bộ giải nén C-Blosc2 cấp phát một vùng đệm đơn liên tục hơn **168 MB**. 
* Do cơ chế quản lý bộ đệm nội bộ của thư viện C-Blosc2 bị phân mảnh (buffer fragmentation), nó sẽ văng ngoại lệ:
  ```text
  RuntimeError: Error while getting the buffer
  ```
  Nếu không được xử lý chuẩn mực, ca bệnh này sẽ bị bỏ sót (skip), vi phạm tính toàn vẹn 197/197 ca của bài báo/luận văn.

#### 2. Tại Sao Các Mô Hình Khác (MedNeXt, nnU-Net, U-Mamba) Không Bị?
* **MedNeXt-L k3 và U-Mamba Bot 3D** trong file script đánh giá đã ưu tiên đọc trực tiếp từ thư mục ảnh gốc `imagesTr` bằng **SimpleITK** (`sitk.ReadImage`). SimpleITK giải nén file `.nii.gz` dạng luồng stream liên tục (stream decompression) nên xử lý trơn tru mọi kích thước thể tích mà không bị giới hạn bộ đệm chunk của Blosc2.
* **Standard nnU-Net và ResEnc-L** sử dụng cơ chế nạp dữ liệu chuẩn của `nnUNetv2_predict` (Memory-mapped Virtual Memory Paging của Linux kernel), hoàn toàn không phụ thuộc thư viện Python Blosc2.
* **SwinUNETR v2** ban đầu bị lỗi là do script cố tình đảo ưu tiên đọc file `.b2nd` lên trước file gốc `.nii.gz` nhằm tối ưu tốc độ đọc đĩa.

#### 3. Quy Chuẩn Đồng Nhất Bắt Buộc Cho Mọi Script Đánh Giá:
* **Quy tắc vàng 1:** Mọi script kiểm thử (`evaluate_*.py`) **bắt buộc phải ưu tiên nhánh đọc ảnh gốc NIfTI (`if has_raw:`) bằng SimpleITK lên hàng đầu**. Việc này vừa miễn nhiễm 100% với lỗi bộ đệm Blosc2, vừa giữ nguyên ma trận toạ độ (Spacing, Origin, Direction) chuẩn giải phẫu để tính toán **HD95** và **ASSD** chính xác tuyệt đối.
* **Quy tắc vàng 2 (Phòng vệ chiều sâu):** Nếu bắt buộc phải nạp từ cache `.b2nd` (khi môi trường không có file `.nii.gz` thô), hàm đọc bắt buộc phải có cơ chế fallback giải nén từng lát Z (`img_h[c, z]`) để bộ đệm mỗi lần chỉ tốn 1 MB, tuyệt đối không được đọc nguyên khối một lần trên các ca ngoại cỡ.

---

### 9.5. 🎯 Quy Chuẩn Bóc Tách 3 Pha (Ablation Study): Tận Dụng 100% Mã Nguồn Python Tùy Biến (Không Cần Tạo Dataset Mới)

#### 1. Nguyên Lý Tự Chủ Mã Nguồn Dự Án (Code Ownership & Flexibility):
* Toàn bộ pipeline huấn luyện trong repository này (`train_*.py`) được xây dựng bằng **mã nguồn Python/PyTorch tự chủ**, trực tiếp quản trị class DataLoader, Patch Sampler và kiến trúc mạng.
* Chúng ta **hoàn toàn KHÔNG bị trói buộc** bởi các hạn chế đóng hộp của các công cụ dòng lệnh (như lệnh CLI thô `nnUNetv2_train` - vốn bị phụ thuộc vào số kênh khai báo tĩnh trong `dataset.json`).

#### 2. Kỹ Thuật Cắt Lát Động Kênh Thời Gian (Dynamic Channel Slicing):
* Bộ dữ liệu tiền xử lý chuẩn hóa `Dataset501` (dù ở định dạng `.b2nd`, `.npy` hay ảnh thô `.nii.gz`) đã lưu trữ sẵn 5 kênh:
  * Kênh 0: Pha trước tiêm (Pre-contrast / T0)
  * Kênh 1: Pha sớm ngấm thuốc (Early Peak / T_early)
  * Kênh 2: Pha muộn thải thuốc (Delayed Late / T_late)
  * Kênh 3: Hiệu vi sai trừ sớm (Sub_early = Early - Pre)
  * Kênh 4: Hiệu vi sai trừ muộn (Sub_late = Late - Pre)
* Khi thực hiện nghiên cứu bóc tách **3 pha tự nhiên** (Pre, Early, Late) theo tiêu chuẩn y sinh của Naomi Fridman (Nature Scientific Data):
  * **TUYỆT ĐỐI KHÔNG CẦN TẠO DATASET MỚI** (như `Dataset502` hay bất kỳ thư mục phụ trợ nào).
  * **TUYỆT ĐỐI KHÔNG CẦN CHẠY LẠI TIỀN XỬ LÝ** (`nnUNetv2_plan_and_preprocess`), tiết kiệm 100% thời gian chờ 3-4 tiếng và 40 GB dung lượng ổ cứng.
  * Chỉ cần điều chỉnh đúng **2 vị trí cốt lõi** trong script Python huấn luyện:
    1. **Khởi tạo mô hình:** Khai báo số kênh đầu vào `in_channels=3` (thay vì 5):
       ```python
       model = ModelArchitecture(in_channels=3, num_classes=2, ...)
       ```
    2. **Hàm Patch Sampler / DataLoader:** Cắt lát lấy 3 kênh đầu tiên `[:3]` trực tiếp từ tensor nạp:
       ```python
       # Lấy đúng 3 kênh 0, 1, 2; hoàn toàn bỏ qua kênh 3 và 4
       patch_img = img_arr[:3, s_0:s_0+p_0, s_1:s_1+p_1, s_2:s_2+p_2]
       ```

#### 3. Minh Chứng Thực Nghiệm Thành Công:
* Minh chứng điển hình nhất là **MedNeXt-L k3 (3 pha)** (`train_mednext_3phases_fold0.py`):
  * Tận dụng 100% dữ liệu có sẵn của `Dataset501`.
  * Chạy mượt mà 200 epochs trong 18h 54m (tăng tốc 40%).
  * Đạt Best Proxy Val Dice **0.7912** và hoàn tất đánh giá 197 ca mà không cần sinh thêm bất kỳ file dữ liệu trung gian nào.
* Quy tắc này được áp dụng nhất quán cho toàn bộ các mô hình tiếp theo trong portfolio: **SwinUNETR v2 (3 pha)**, **Vanilla 3D U-Net (3 pha)**, **PrimusV2 (3 pha)** và **U-Mamba (3 pha)**.
