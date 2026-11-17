# Hướng Dẫn & Quy Tắc Sử Dụng Máy Chủ / GPU Server Trường
# (University GPU Cluster & Lab Server Guidelines)

Tài liệu này tổng hợp toàn bộ các **nguyên tắc bắt buộc từ Thầy hướng dẫn** và **quy trình kỹ thuật chuẩn** khi SSH vào máy chủ trường / phòng lab (RTX 4090 / 4080) để chạy các mô hình 3D Segmentation (MedNeXt-L k3, U-Mamba Bot 3D).

---

## 📋 PHẦN 1: CÁC NGUYÊN TẮC CỐT LÕI TỪ THẦY HƯỚNG DẪN

### 0. ⛔ NGUYÊN TẮC BẤT KHẢ XÂM PHẠM: TUYỆT ĐỐI KHÔNG GIẢI PHÓNG Ổ ĐĨA & KHÔNG ĐỤNG CHẠM DỮ LIỆU HIỆN CÓ TRÊN MÁY TRƯỜNG
* **Cam kết bảo toàn hệ thống:**
  * **Tuyệt đối KHÔNG chạy các lệnh giải phóng ổ đĩa**: Không dùng `rm`, `clean`, `purge`, `apt-get clean`, `conda clean`, `find ... -delete`.
  * **Toàn bộ dữ liệu 350 GB hiện có** của các thành viên khác trong phòng lab trên phân vùng `/dev/nvme1n1p3` được **bảo toàn nguyên vẹn 100%**, không di dời, không sửa đổi, không xóa bỏ dù chỉ 1 file.
  * **Chỉ hoạt động trong phạm vi thư mục riêng**: Toàn bộ code, môi trường ảo `venv`, dữ liệu và log chỉ được tạo và lưu trữ độc lập bên trong thư mục tài khoản cá nhân của bạn (`~/breastcancer_segmentation`), không bao giờ ghi hay can thiệp ra ngoài.
  * **Kiểm soát dung lượng an toàn trong 82 GB còn trống**:
    * Chỉ kéo đúng dữ liệu preprocessed cần thiết (không kéo thừa).
    * Code huấn luyện chỉ lưu duy nhất 2 file trọng số: `checkpoint_best.pth` và `checkpoint_final.pth` (~250 MB mỗi file), tuyệt đối không lưu checkpoint theo từng epoch làm phình ổ cứng.

### 0.1 ⛔ NGUYÊN TẮC TOÀN VẸN MÃ NGUỒN: BẮT BUỘC DÙNG 100% MÃ NGUỒN GỐC CÔNG BỐ, CẤM TỰ CHẾ
* **Yêu cầu bắt buộc:**
  * Mọi mô hình (MedNeXt, SwinUNETR, Primus, U-Mamba, SegMamba) **BẮT BUỘC phải lấy trực tiếp từ repository chính thức của tác giả công bố** hoặc thư viện chuẩn quốc tế (MONAI, nnU-Net, dynamic-network-architectures).
  * **Tuyệt đối CẤM tự ý viết lại (re-implement), tự ý đơn giản hóa hay tự chế kiến trúc**.
  * Chỉ được phép sửa đúng 2 tham số: `in_channels` (5 pha DCE-MRI) và `num_classes` (2 lớp Background, Tumor).
* **Lý do an toàn máy chủ & học thuật:**
  * Các bản code "tự chế" tiềm ẩn nguy cơ bùng nổ chiều batch (như lỗi batch 32.768 trong SegMamba nháp) gây tràn RAM, kích hoạt Linux OOM-Killer làm sập toàn bộ máy chủ trường và ảnh hưởng đến các thành viên khác trong phòng lab.
  * Làm sai lệch bản chất mô hình khiến kết quả trong luận văn/bài báo bị bác bỏ hoàn toàn.

### 0.2 ⛔ NGUYÊN TẮC MẠNG: SERVER KHÔNG KẾT NỐI GITHUB – CÁCH ĐỒNG BỘ MÃ NGUỒN
* **Thực tế hạ tầng:**
  * Server trường `node1` **hoàn toàn KHÔNG kết nối tài khoản GitHub**, không có git credentials/ssh keys cho GitHub.
  * Mọi thao tác `git pull` hay `git clone` trên server trường đều sẽ thất bại.
* **Quy chuẩn đồng bộ chuẩn xác (2 phương pháp):**

#### Phương pháp 1: Kéo trực tiếp ngay trên terminal `node1` qua `curl` (Khuyên dùng - Nhanh nhất)
Vì GitHub repository là Public, bạn chạy trực tiếp các lệnh sau ngay tại terminal của `node1` (`dung@node1:~/breastcancer_segmentation$`):
```bash
cd ~/breastcancer_segmentation

# Kéo file SwinUNETR v2 3-pha (đã fix buffer):
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/train_swinunetr_v2_3phases_fold0.py -o scripts/train_swinunetr_v2_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/run_swinunetr_v2_3phases_server.sh -o scripts/run_swinunetr_v2_3phases_server.sh

# Kéo các script 3-pha cho các mô hình tiếp theo (U-Mamba, SwinUNETR v1, PrimusV2):
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/train_umamba_3phases_fold0.py -o scripts/train_umamba_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/evaluate_umamba_3phases_fold0.py -o scripts/evaluate_umamba_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/run_umamba_3phases_server.sh -o scripts/run_umamba_3phases_server.sh

curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/train_swinunetr_3phases_fold0.py -o scripts/train_swinunetr_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/evaluate_swinunetr_3phases_fold0.py -o scripts/evaluate_swinunetr_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/run_swinunetr_v1_3phases_server.sh -o scripts/run_swinunetr_v1_3phases_server.sh

curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/train_primus_v2_3phases_fold0.py -o scripts/train_primus_v2_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/evaluate_primus_v2_3phases_fold0.py -o scripts/evaluate_primus_v2_3phases_fold0.py
curl -sSL https://raw.githubusercontent.com/XiaoSha59/breastcancer_segmentation/main/scripts/run_primus_v2_3phases_server.sh -o scripts/run_primus_v2_3phases_server.sh

chmod +x scripts/*.sh
```

#### Phương pháp 2: Đẩy từ máy cá nhân Windows qua SCP
Chạy trên terminal PowerShell tại máy Windows (`D:\breastcancer_segmentation`):
```powershell
scp scripts/train_swinunetr_v2_3phases_fold0.py dung@node1:~/breastcancer_segmentation/scripts/
scp scripts/run_swinunetr_v2_3phases_server.sh dung@node1:~/breastcancer_segmentation/scripts/
```


### 1. 🚫 CẤM DÙNG NOTEBOOK (`.ipynb`) – 100% CHẠY QUA TERMINAL SCRIPT (`.py`)
* **Lý do:**
  * Jupyter Notebook giữ chặt toàn bộ VRAM GPU ngay cả khi cell chạy xong, khiến các thành viên khác trong phòng lab không có tài nguyên để dùng.
  * Khó kiểm soát log dài ngày, dễ bị tràn RAM/VRAM ngầm, và không đảm bảo tính tái lập (reproducibility) của nghiên cứu khoa học.
* **Quy tắc:**
  * Toàn bộ mã nguồn phải ở định dạng file Python thuần `.py` hoặc Shell script `.sh`.
  * Tuyệt đối không để kernel Jupyter chạy ngầm chiếm card.

### 2. 🛡️ BẮT BUỘC CHẠY TRONG `tmux` (CHỐNG RỚT MẠNG & GẬP MÁY)
* **Lý do:**
  * Mạng WiFi trường rất dễ chập chờn hoặc timeout. Nếu chạy trực tiếp trên terminal SSH thường, khi bạn gập laptop hoặc mạng đứt, Linux sẽ gửi tín hiệu `SIGHUP` tiêu diệt ngay lập tức tiến trình huấn luyện.
* **Quy trình chuẩn:**
  ```bash
  # Tạo session mới:
  tmux new -s train_mednext

  # Chạy script bên trong tmux:
  bash scripts/run_mednext_fold0.sh

  # Thoát tạm thời (Detach) an toàn:
  # Nhấn: Ctrl + B, sau đó thả tay và nhấn phím D

  # Gắn lại vào session (Attach) khi SSH lại:
  tmux attach -t train_mednext
  ```

### 3. 🎯 KIỂM TRA & CHỈ ĐỊNH RÕ RÀNG GPU (`CUDA_VISIBLE_DEVICES`)
* **Lý do:**
  * Máy chủ lab là tài nguyên dùng chung. Tuyệt đối không để script mặc định chiếm tất cả GPU hoặc ghi đè lên GPU người khác đang train.
* **Quy tắc:**
  * Luôn chạy `nvidia-smi` để tìm card trống (`Memory-Usage` = 0 MiB hoặc < 500 MiB).
  * Luôn gán rõ ràng GPU được phép sử dụng:
    ```bash
    export CUDA_VISIBLE_DEVICES=0  # Hoặc 1, 2 tùy theo card nào còn trống
    ```

### 4. 💾 QUẢN LÝ Ổ ĐĨA & QUY TẮC BUFFER AN TOÀN TRÁNH TREO MÁY
* **Lý do:**
  * Phân vùng `/home/username` trên máy trường thường nằm chung trên ổ đĩa hệ thống `/` (chỉ có 80 GB – 100 GB trống).
  * Bộ dữ liệu preprocessed (`Dataset501`) nặng khoảng **~64 GB**. Nếu tải trực tiếp vào `~`, nó sẽ vét sạch dung lượng trống, làm ổ cứng chạm 100% khiến hệ điều hành đóng băng cổng đăng nhập SSH (`Entering interactive session`).
* **Quy tắc Buffer Ổ Đĩa (Disk Buffer):**
  * **Mức Buffer Tối Thiểu:** Luôn phải chừa tối thiểu **`25 GB – 30 GB` dung lượng trống bất khả xâm phạm** cho phân vùng hệ điều hành (`/`) để Linux lưu file log, swap, session và file tạm của các user khác.
  * **Công thức an toàn:** `Dung lượng còn trống (Avail)` $\ge$ `Dung lượng dữ liệu (64 GB) + 30 GB Buffer` $\rightarrow$ **Cần tối thiểu 95 GB trống**.
  * Nếu ổ hiện tại có `Avail < 95 GB`: **Tuyệt đối KHÔNG lưu vào thư mục Home (`~`)**. Phải tìm phân vùng lưu trữ lớn (`/home/students`, `/data`, `/scratch`, `/mnt/storage` - nơi thường có hàng TB trống) và dùng symbolic link (`ln -s`) trỏ về project.

### 4.1. ⚡ QUY TẮC BUFFER AN TOÀN CHO GPU VRAM (TRÁNH CRASH DRIVER & OOM)
* **Lý do:**
  * Khi GPU bị ép sát trần 100% VRAM (ví dụ 15.9 GB / 16 GB trên RTX 4080, hoặc 23.9 GB / 24 GB trên RTX 4090), các đột biến bộ nhớ bất ngờ trong bước backward pass hoặc sliding window validation sẽ dẫn đến lỗi CUDA OOM hoặc nguy hiểm hơn là làm nghẽn bus PCIe / treo Driver NVIDIA (GPU Lock), kéo theo tê liệt máy chủ lab.
* **Mức Buffer VRAM Chuẩn:**
  * **RTX 4080 (16 GB VRAM):** Chỉ cấp phát tối đa **`12 GB – 13 GB`**, luôn chừa **`3 GB – 4 GB Buffer`**.
  * **RTX 4090 (24 GB VRAM):** Chỉ cấp phát tối đa **`18 GB – 20 GB`**, luôn chừa **`4 GB – 5 GB Buffer`**.
* **Kỹ thuật bắt buộc trong mã nguồn để đảm bảo Buffer:**
  1. **Bật AMP Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`)**: Giảm 50% dung lượng VRAM lưu activations và weights.
  2. **Bật Gradient Checkpointing (`use_checkpoint=True`)**: Giảm thêm 30-40% VRAM đỉnh mà chỉ tốn thêm ~10% thời gian tính toán.
  3. **Giới hạn Sliding Window Batch:** Khi validate 197 ca, đặt `sw_batch_size=2` hoặc `4`, `overlap=0.5` để không làm bùng nổ bộ nhớ.

### 5. 📜 GHI LOG LIÊN TỤC VÀO FILE (`train.log`)
* **Lý do:**
  * Quá trình huấn luyện kéo dài 200 epochs (nhiều giờ/ngày), cần có bằng chứng ghi nhận tiến độ, kiểm tra loss, proxy dice và bắt lỗi nếu xảy ra sự cố.
* **Quy tắc:**
  * Runner script đã được cấu hình tự động lưu log vào file `results/.../train.log`.
  * Theo dõi tiến độ từ xa không cần mở tmux:
    ```bash
    tail -n 25 results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0/train.log
    tail -f results/.../train.log  # Theo dõi trực tiếp theo thời gian thực (Ctrl+C để thoát)
    ```

### 6. 🔒 MÔI TRƯỜNG ẢO PYTHON VENV (`breast_env`), TUYỆT ĐỐI KHÔNG DÙNG CONDA
* **Thực tế hạ tầng:**
  * Toàn bộ thư viện của dự án được cài đặt trong môi trường ảo **Python `venv`** có tên là **`breast_env`**, dự án **KHÔNG SỬ DỤNG CONDA**.
* **Lệnh kích hoạt chuẩn**:
  ```bash
  source ~/breast_env/bin/activate
  ```
  *(Khi thấy đầu dòng terminal xuất hiện `(breast_env)` là đã nạp đầy đủ PyTorch, CUDA, MONAI và các thư viện cần thiết).*


### 7. ⚠️ TUYỆT ĐỐI KHÔNG CAN THIỆP TẮT MÁY / REBOOT SERVER
* **Lý do:**
  * Server trường phục vụ nghiên cứu liên tục 24/7 của cả khoa/viện.
  * Các lệnh tự động tắt máy (`sudo shutdown -h now`) vốn chỉ dùng riêng cho máy ảo GCP Cloud tính tiền theo giờ. **Trên server trường, đây là điều cấm kỵ.**
* **Bảo đảm an toàn trong code:**
  * Các file `scripts/run_mednext_fold0.sh` và `scripts/run_umamba_fold0.sh` đã được gỡ bỏ 100% lệnh tắt máy. Sau khi train và evaluate xong 197 ca, server sẽ giữ nguyên trạng thái hoạt động bình thường.

---

## 🚀 PHẦN 2: CHECKLIST CÁC LỆNH KIỂM TRA TRƯỚC KHI CHẠY (PRE-FLIGHT)

Mỗi lần SSH vào máy trường, bạn hãy thực hiện tuần tự 6 bước sau:

### Bước 1: Kiểm tra phần cứng GPU
```bash
nvidia-smi
```
* **Mục tiêu:** Xác nhận card là **RTX 4090 (24GB)** hoặc **RTX 4080 (16GB)**.
* **Lưu ý:** Xem cột **Processes**: nếu có process của người khác đang chiếm card, bạn phải chọn card khác (`export CUDA_VISIBLE_DEVICES=X`).

### Bước 2: Kiểm tra dung lượng ổ đĩa
```bash
df -h .
```
* **Mục tiêu:** Đảm bảo phân vùng làm việc còn trống tối thiểu **100 GB** (cho 64 GB data preprocessed + 10 GB weights/checkpoints + log).

### Bước 3: Kiểm tra RAM hệ thống và CPU
```bash
free -h
nproc
```
* **Mục tiêu:** Đảm bảo RAM hệ thống còn trống $\ge$ **16 - 32 GB** để DataLoader đọc nhanh các file `.b2nd`.

### Bước 4: Kiểm tra PyTorch & CUDA Driver
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')"
```
* **Kỳ vọng:** In ra `CUDA Available: True`, kèm tên GPU và VRAM chuẩn xác.

### Bước 5: Kiểm tra các thư viện bắt buộc
```bash
python3 -c "import blosc2, monai, einops; print('Dependencies OK!')"
```
* Nếu thiếu, cài đặt nhanh:
  ```bash
  pip install --user blosc2 monai einops
  ```

### Bước 6: Chạy thử Unit-Test 30 giây (Dry-Run xác nhận pipeline)
```bash
python3 scripts/test_mednext_pipeline.py
```
* Script này sẽ test nhanh 1 lượt Forward, Backward, Mixed Precision FP16, Gradient Checkpointing và Sliding Window Inference giả lập.
* Khi terminal in ra:
  ```text
  Peak allocated VRAM: 9.69 GB
  === ALL CHECKS PASSED SUCCESSFULLY! ===
  ```
  bạn hoàn toàn yên tâm mô hình sẽ chạy trơn tru mà không bị lỗi CUDA OOM.

---

## 📦 PHẦN 3: KÉO DỮ LIỆU PREPROCESSED VỀ SERVER TRƯỜNG

Không cần tiền xử lý lại từ đầu (mất 5-6 tiếng). Sử dụng script hỗ trợ đã tích hợp sẵn:
```bash
bash scripts/pull_data_to_cluster.sh
```
Script sẽ cung cấp 3 lựa chọn:
1. **Lựa chọn 1 (Nhanh nhất):** SCP/rsync trực tiếp từ máy ảo GCP (`34.87.101.121`) nếu máy ảo vẫn đang mở.
2. **Lựa chọn 2:** Kéo trực tiếp từ Kaggle Datasets (tốc độ mạng trường gigabit ~10-15 phút).
3. **Lựa chọn 3:** Kéo từ Google Drive thông qua công cụ `gdown`.

---

## 🏃 PHẦN 4: KHỞI CHẠY HUẤN LUYỆN MEDNEXT-L K3 TRÊN MÁY TRƯỜNG

```bash
# 1. Mở session tmux
tmux new -s mednext

# 2. Khởi chạy runner script
bash scripts/run_mednext_fold0.sh

# 3. Thoát tmux an toàn:
# Nhấn Ctrl + B, sau đó nhấn D

# 4. Khi cần xem lại tiến độ:
tmux attach -t mednext
# Hoặc xem 30 dòng log mới nhất:
tail -n 30 results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0/train.log
```
