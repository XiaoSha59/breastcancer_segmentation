# Hướng Dẫn Sử Dụng & Chuẩn Hóa Dữ Liệu DCE-MRI (Dataset Guide)

> **Mục tiêu:** Tài liệu chuẩn hóa toàn diện quy trình tải, giải nén, lọc pha và xây dựng tensor 5 kênh cho bài toán **Phân vùng khối u 3D DCE-MRI kết hợp phân loại thụ thể HER2 (Multi-Task Learning)**. Đảm bảo tính nhất quán và công bằng học thuật (Fair Comparison) giữa các công bố của **Naomi Fridman et al.** và **Elena Krikun et al.**

---

## 1. TỔNG QUAN VỀ CÁC BỘ DỮ LIỆU (COHORT OVERVIEW)

| Đặc tính | Bộ Huấn luyện & Đánh giá nội bộ (Development) | Bộ Kiểm định ngoại kiểm (External Test) |
| :--- | :--- | :--- |
| **Tên Cohort** | **I-SPY2 (BreastDCEDL_ISPY2)** | **I-SPY1 (BreastDCEDL_ISPY1)** |
| **Cỡ mẫu ($n$)** | **982 bệnh nhân** | **171 bệnh nhân** |
| **Mục đích** | Huấn luyện 5-fold Cross-Validation & đối đầu trực tiếp với Baseline B1 | Đánh giá tổng quát hóa Zero-Shot (không fine-tune) |
| **Nguồn phát hành** | The Cancer Imaging Archive (TCIA) / Zenodo | Zenodo Record `18114231` |
| **Dung lượng thô** | ~54 GB (gồm 3–12 thời điểm DCE) | ~1.17 GB (bản tinh gọn) |
| **Dung lượng sau lọc**| **~15 GB** (chỉ giữ 3 pha chuẩn $I_0, I_e, I_l$) | **~1.17 GB** |

---

## 2. NGUỒN TẢI & HƯỚNG DẪN TẢI DỮ LIỆU

### 2.1. Bộ I-SPY2 (Gói Full 54 GB từ TCIA)
* **Trang thông tin:** [TCIA BreastDCEDL-ISPY2](https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/) (DOI: `10.7937/42WQ-TH78`)
* **Package Context URL:**
  ```text
  https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjExNTgiLCJwYXNzY29kZSI6IjVjZTYxMmQ3MmE2MTE1MDE0ZDRiZjdkNWRlYjI0OThiOWMzZDg1M2MiLCJwYWNrYWdlX2lkIjoiMTE1OCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ==
  ```

#### ⚡ Phương án 1: Tải bằng Aspera CLI (`ascli`) - KHUYÊN DÙNG (ỔN ĐỊNH 100%)
Do giao diện web và app Aspera Connect thường gặp lỗi kết nối (chặn cổng UDP hoặc lỗi bắt tay trình duyệt `Unable to download`), giải pháp chuẩn kỹ thuật của TCIA là sử dụng công cụ dòng lệnh **Aspera CLI (`ascli`)**:

1. **Tải bộ Aspera CLI Portable (không cần cài đặt, không cần Ruby):**
   * [aspera-cli-4.28.0-windows-x86_64-portable.zip](https://github.com/IBM/aspera-cli/releases/download/v4.28.0/aspera-cli-4.28.0-windows-x86_64-portable.zip)
   * Giải nén vào thư mục `tools/ascli/` trong project.

2. **Lệnh tải toàn bộ 54 GB về thư mục `data/raw` (PowerShell):**
   ```powershell
   & ".\tools\ascli\ascli.cmd" faspex5 packages receive `
     --url="https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjExNTgiLCJwYXNzY29kZSI6IjVjZTYxMmQ3MmE2MTE1MDE0ZDRiZjdkNWRlYjI0OThiOWMzZDg1M2MiLCJwYWNrYWdlX2lkIjoiMTE1OCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ==" `
     --to-folder="D:\breastcancer_segmentation\data\raw"
   ```

3. **Lệnh tải thử nghiệm 1 ca bệnh cụ thể (Ví dụ: `ACRIN-6698-102212`):**
   ```powershell
   & ".\tools\ascli\ascli.cmd" faspex5 packages receive `
     --url="<URL_GÓI_TRÊN>" `
     --to-folder="D:\breastcancer_segmentation\data\raw" `
     "BreastDCEDL_ISPY2/ACRIN-6698-102212"
   ```

> [!TIP]
> **Đặc tính của `ascli`:** Chạy trực tiếp qua giao thức truyền tải FASP tối ưu băng thông. Nếu quá trình tải bị ngắt giữa chừng, chỉ cần chạy lại lệnh, hệ thống sẽ tự động đối chiếu và bỏ qua các file đã tải xong mà không tải lại từ đầu.

#### 🌐 Phương án 2: Tải qua trình duyệt (Aspera Connect Web GUI)
Nếu sử dụng giao diện web:
1. Mở link gói trên bằng trình duyệt Chrome/Edge.
2. Cài đặt tiện ích mở rộng **IBM Aspera Connect** trên Chrome Web Store và app Desktop.
3. Trong cài đặt app Aspera (icon ⚙️) $\to$ **Transfers** $\to$ bật tùy chọn **"HTTP fallback"** (tránh bị lỗi mạng chặn cổng UDP).
4. Bấm **Download all** hoặc double click vào thư mục `BreastDCEDL_ISPY2` để tải từng phần.

### 2.2. Bộ I-SPY1 External Test (1.17 GB từ Zenodo)
* **DOI:** `10.5281/zenodo.18114231` (Record mới nhất của Naomi Fridman).
* **Link tải trực tiếp:** [BreastDCEDL_ISPY1_min_crop.tar.gz (1.17 GB)](https://zenodo.org/api/records/18114231/files/BreastDCEDL_ISPY1_min_crop.tar.gz/content)
* Có thể tải trực tiếp trên máy local hoặc chạy `wget` thẳng về GCP VM.

---

## 3. PHÂN ĐỊNH HỌC THUẬT: NAOMI FRIDMAN vs. ELENA KRIKUN

Một trong những sai lầm phổ biến nhất trong nghiên cứu thực nghiệm là nhầm lẫn chỉ số pha (phase indices) giữa các công bố dẫn đến kết quả không thể so sánh công bằng.

### 3.1. Bảng so sánh chỉ số thời điểm (Phase Indices)

```text
Chuỗi DCE-MRI theo trục thời gian (t = 0 đến 7):
  t=0 (Trước tiêm)   t=1 (Ngay sau tiêm)   t=2 (~90-120s ngấm đỉnh)   ...   t=6,7 (~420s thải thuốc)
   │                  │                    │                                 │
   ▼                  ▼                    ▼                                 ▼
   I_0               I_e (Krikun)         I_e (Naomi)                       I_l (Krikun / Naomi)
```

| Tác giả & Tài liệu | Pha Pre ($I_0$) | Pha Early ($I_e$) | Pha Late ($I_l$) | Vai trò trong dự án của chúng ta |
| :--- | :--- | :--- | :--- | :--- |
| **Elena Krikun et al. (EJR AI 2026)**<br>File: `references/metadata/cases_all.csv` | **Pha 0** | **Pha 1** (first post-contrast) | **Pha cuối** (Pha 6 hoặc 7) | **CHỈ TIÊU CỐT LÕI:** Dùng để lọc ảnh NIfTI và chia đúng 5-fold CV để so khớp công bằng với Baseline B1 (Dice 0.778). |
| **Naomi Fridman et al. (Nature Sci Data 2026)**<br>File: `references/metadata/BreastDCEDL_ISPY2_metadata.csv` | **Pha 0** | **Pha 2** (peak enhancement) | **Pha 5 hoặc 6** (washout) | **NGUỒN NHÃN SINH HỌC:** Cung cấp nhãn phân tử chuẩn hóa (`HER2`, `HR`, `pCR`) cho nhánh phân loại phụ Multi-Task. |

> [!IMPORTANT]
> **Quy tắc bất di bất dịch:** Để tái lập đúng điểm số Segmentation và chứng minh mô hình Multi-task của chúng ta vượt trội hơn Baseline B1 của Krikun, chúng ta **bắt buộc phải trích xuất các pha hình ảnh theo đúng chỉ số trong `cases_all.csv`** của Krikun. Nhãn HER2 được lấy từ bảng của Naomi map qua mã định danh bệnh nhân (`pid`).

---

## 4. CẤU TRÚC TENSOR ĐẦU VÀO 5 KÊNH DƯỢC ĐỘNG HỌC ($C=5$)

Theo nguyên lý mô hình hóa động học ngấm thuốc mô ung thư (Pharmacokinetics), các kênh tensor đầu vào được thiết lập như sau:

$$\mathbf{X} \in \mathbb{R}^{5 \times D \times H \times W}$$

1. **Kênh 0 ($I_0$):** Ảnh gốc trước khi tiêm thuốc đối quang từ (Pre-contrast). Phản ánh giải phẫu nền của mô tuyến vú.
2. **Kênh 1 ($I_e$):** Ảnh pha sớm sau tiêm (Early post-contrast, $t \approx 90\text{s}$). Phản ánh sự tăng sinh mạch máu bất thường (Angiogenesis) của khối u ác tính.
3. **Kênh 2 ($I_l$):** Ảnh pha muộn sau tiêm (Late post-contrast, $t \approx 420\text{s}$). Phản ánh động học rửa thuốc (Washout kinetics).
4. **Kênh 3 ($\Delta I_e = I_e - I_0$):** Ảnh trừ pha sớm (Early subtraction). Triệt tiêu toàn bộ tín hiệu mỡ và mô nền, làm nổi bật viền khối u.
5. **Kênh 4 ($\Delta I_l = I_l - I_0$):** Ảnh trừ pha muộn (Late subtraction). Làm rõ mức độ giữ/thải thuốc ở trung tâm u so với viền.

* **Tính toán động (Dynamic Computation):** Kênh 3 và Kênh 4 được tính toán tức thời trong bộ nhớ RAM của `DataLoader` bằng phép trừ vector (`subtraction = early - pre`), không lưu trữ trên ổ đĩa để tiết kiệm 40% dung lượng.

---

## 5. QUY TRÌNH THU GỌN DỮ LIỆU: TỪ 54 GB XUỐNG 15 GB (DATA SLIMMING)

Gói 54 GB thô chứa từ 7 đến 12 pha cho mỗi bệnh nhân. Quy trình thu gọn được thực hiện tự động qua script:

```text
Gói thô TCIA (54 GB, 8.021 file .nii.gz)
   │
   ▼ Đọc references/metadata/cases_all.csv
   ├─ Giữ lại: pre_path (Pha 0)
   ├─ Giữ lại: early_post_path (Pha 1)
   ├─ Giữ lại: late_post_path (Pha cuối)
   ├─ Giữ lại: mask_path (Mask khối u 3D)
   └─ Xóa bỏ: Các pha trung gian dư thừa (Pha 2, 3, 4, 5...)
   │
   ▼
Bộ dữ liệu chuẩn hóa 15 GB (982 bệnh nhân × 3 pha + 1 mask)
```

---

## 6. CẤU TRÚC THƯ MỤC CHUẨN TRÊN LOCAL & VM

Sau khi lọc, cấu trúc thư mục trong project sẽ có dạng:

```text
breastcancer_segmentation/
├── data/
│   ├── raw/                             # Thư mục chứa gói giải nén 54 GB ban đầu
│   ├── processed/                       # Dữ liệu tinh gọn 15 GB đã trích xuất 3 pha
│   │   ├── imagesTr/                    # Ảnh 3 pha của 982 ca I-SPY2
│   │   │   ├── ACRIN-6698-102212_0000.nii.gz   # I_0
│   │   │   ├── ACRIN-6698-102212_0001.nii.gz   # I_e
│   │   │   └── ACRIN-6698-102212_0002.nii.gz   # I_l
│   │   ├── labelsTr/                    # Mask nhãn khối u 3D
│   │   │   └── ACRIN-6698-102212.nii.gz
│   │   └── splits_final.json            # 5-fold CV map chuẩn từ cases_all.csv
│   └── external/                        # Cohort ngoại kiểm I-SPY1 (171 ca)
├── references/
│   └── metadata/
│       ├── cases_all.csv                # Manifest chuẩn của Elena Krikun (5 folds, phase paths)
│       └── BreastDCEDL_ISPY2_metadata.csv # Manifest nhãn phân tử của Naomi Fridman (HER2/HR/pCR)
└── scripts/
    ├── 01_extract_3phases.py            # Script lọc và đóng gói 15 GB
    └── 02_verify_dataset_integrity.py   # Script kiểm tra toàn vẹn dữ liệu
```

---

## 7. CHECKLIST KIỂM TRA TOÀN VẸN (INTEGRITY QA CHECKLIST)

Trước khi bắt đầu huấn luyện bất kỳ mô hình nào, script kiểm định phải xác nhận đạt 100% các tiêu chí sau:

1. [ ] **Đủ số lượng:** Đúng 982 ca bệnh I-SPY2, không thiếu bất kỳ ca nào so với `cases_all.csv`.
2. [ ] **Toàn vẹn NIfTI:** Header NIfTI đọc được bình thường, `affine` matrix giữa 3 pha và Mask phải khớp nhau hoàn toàn.
3. [ ] **Không có Mask rỗng:** Mọi mask nhãn phải chứa ít nhất 1 voxel dương tính ($V_{\text{tumor}} > 0$).
4. [ ] **Khớp nhãn HER2:** 982 ca bệnh đều được gán nhãn HER2 hợp lệ ($0.0$ hoặc $1.0$) từ bảng Naomi.
5. [ ] **Phân chia Fold công bằng:** 5-fold CV khớp tuyệt đối với cột `fold` trong `cases_all.csv`.
