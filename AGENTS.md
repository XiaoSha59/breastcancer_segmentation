# AGENTS.md - Behavioral Guidelines & Strict Engineering Rules for AI Assistants

## ⛔ NGUYÊN TẮC BẤT KHẢ XÂM PHẠM: BẮT BUỘC DÙNG 100% MÃ NGUỒN GỐC CÔNG BỐ, TUYỆT ĐỐI CẤM TỰ CHẾ

Khi thực hiện bất kỳ tác vụ lập trình, huấn luyện, đánh giá hay so sánh mô hình trong repository này:
1. **BẮT BUỘC DÙNG MÃ NGUỒN GỐC TỪ TÁC GIẢ CÔNG BỐ**:
   - Mọi mô hình (MedNeXt, SwinUNETR, Primus, U-Mamba, SegMamba,...) **phải được lấy trực tiếp 100% nguyên bản** từ repository chính thức của nhóm tác giả (đã công bố tại MICCAI, CVPR, ICCV, Nature/IEEE) hoặc từ các thư viện chuẩn quốc tế (MONAI, nnU-Net, dynamic-network-architectures).
   - Trước khi đưa code vào dự án, AI assistant **bắt buộc phải cung cấp URL mã nguồn gốc** của tác giả để đối chiếu.

2. **TUYỆT ĐỐI CẤM TỰ Ý VIẾT LẠI HOẶC TỰ CHẾ KIẾN TRÚC**:
   - Nghiêm cấm mọi hành vi tự ý "đơn giản hóa", tự viết lại class mô hình thay cho mã nguồn của tác giả.
   - Không được tự ý thay đổi cơ chế quét token, cơ chế trích xuất đặc trưng cục bộ (như GSC trong SegMamba, Inverted Bottleneck trong MedNeXt, Shifted Window trong SwinUNETR).
   - Chỉ được phép điều chỉnh đúng 2 tham số đầu vào/ra của bài toán: `in_channels` (5 pha DCE-MRI) và `num_classes` (2 lớp).

3. **LÝ DO KỸ THUẬT & HỌC THUẬT**:
   - Mã nguồn tự chế tiềm ẩn nguy cơ sai lệch đồ thị tính toán Autograd, bùng nổ kích thước batch gây tràn VRAM / RAM và làm sập máy chủ dùng chung của trường đại học.
   - Đảm bảo tính trung thực và hợp lệ khoa học tuyệt đối cho luận văn / bài báo.

## 🛡️ CÁC NGUYÊN TẮC AN TOÀN TRÊN SERVER TRƯỜNG (`node1`)
1. **Không tự ý shutdown (Zero auto-shutdown)**: Máy chủ trường chạy 24/7 cho cả phòng lab, nghiêm cấm mọi lệnh tắt máy (`shutdown`, `poweroff`, `halt`).
2. **Không giải phóng ổ đĩa / Không xóa dữ liệu**: Tuyệt đối không xóa bất kỳ file nào ngoài thư mục của project.
3. **Giới hạn tài nguyên**: Luôn giới hạn `torch.set_num_threads(4)` và kiểm tra `nvidia-smi` trước khi chạy GPU.
4. **Bảo toàn splits**: Tuyệt đối không sinh lại `splits_final.json`, luôn dùng cố định 785 train / 197 val.
5. **CẤM DÙNG GIT TRÊN SERVER TRƯỜNG (`node1`)**: Máy chủ `node1` **hoàn toàn KHÔNG kết nối tài khoản GitHub**, không thể và không được phép chạy `git pull` hay `git clone`. Mọi file mã nguồn được đồng bộ trực tiếp qua SCP hoặc do người dùng tự tải vào server. Tuyệt đối cấm AI hướng dẫn chạy lệnh git trên máy trường.
6. **CẤM DÙNG JUPYTER NOTEBOOK TRÊN SERVER TRƯỜNG**: Mọi tác vụ huấn luyện và đánh giá bắt buộc phải chạy qua terminal/script bash trong `tmux`, tuyệt đối không dùng file `.ipynb`.

