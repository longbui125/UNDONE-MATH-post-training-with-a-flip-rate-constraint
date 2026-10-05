# GRPO với ràng buộc tỷ lệ đúng → sai

Project chỉ giữ lượt **seed 42, 128 bước, 640 bài test** đã hoàn tất ngày 05/10/2026. Hai phương pháp: GRPO và GRPO + ràng buộc mềm theo chủ đề. Model: **Qwen/Qwen2.5-1.5B-Instruct**.

Báo cáo ngắn (2 trang): [Bài toán, phương pháp, kết quả và hạn chế của lượt run](output/pdf/bao_cao_grpo_rang_buoc_seed42.pdf).

## Bài toán

Hậu huấn luyện model trên nhiều chủ đề toán có thể sửa được các bài vốn sai nhưng cũng làm sai các bài model ban đầu giải đúng. Mục tiêu thử nghiệm là giảm tỷ lệ **đúng → sai** mà vẫn cho model học thêm, đồng thời đo chi phí phát sinh.

## Phương pháp

GRPO sinh 4 lời giải cho một đề, chấm reward 0/1 bằng đáp án nguyên cuối trong `\boxed{...}`, rồi chuẩn hóa reward trong group thành advantage. Objective dùng clipped token ratio; mean trên token mỗi lời giải rồi mean trên group. Mỗi group có 2 policy epochs. Group có reward đồng nhất không có tín hiệu advantage, nên GRPO không cập nhật ở group đó.

Nhánh ràng buộc kiểm tra thêm một anchor mỗi bước. Anchor thuộc tập riêng, được chọn vì **model gốc giải đúng**, không lấy từ validation/test. Với mỗi chủ đề, đếm số lần anchor chuyển thành sai trong cửa sổ 32 bước:

```text
flip_rate_topic = wrong_anchor_checks / total_anchor_checks
lambda_topic = clip(lambda_topic + 0.30 * (flip_rate_topic - 0.10), 0, 1)
loss = GRPO_loss + lambda_topic * gold_solution_loss(anchor)
```

Lambda khởi tạo 0, cập nhật **sau** mỗi 32 bước và dùng từ cửa sổ tiếp theo. Cập nhật sau bước 128 không còn bước train để dùng. Gold loss là negative mean log-probability trên token lời giải mẫu và EOS, không tính token đề bài. Khi lambda > 0, anchor loss vẫn có thể tạo optimizer update dù group RL không có advantage. Vì vậy hai nhánh có cùng ngân sách sampling nhưng không nhất thiết cùng số optimizer update hay tổng compute.

Đây là ràng buộc **mềm**, dùng gold loss làm surrogate để giữ khả năng giải; không bảo đảm mọi đáp án test sẽ không đổi thành sai. Scan anchor, audit model train mới và đánh giá dùng cùng cách chuẩn bị QLoRA để tránh khác biệt precision trước train.

## Thiết lập đã chạy

| Thành phần | Thiết lập |
|---|---|
| Dataset | `HuggingFaceH4/MATH`, revision `9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9` |
| Chủ đề | Đại số, hình học, lý thuyết số, tổ hợp và xác suất |
| Seed | 42 |
| Train / anchor ứng viên / validation / test | 128 / 512 / 160 / 640 |
| Anchor model gốc giải đúng | 153: 64 / 22 / 38 / 29 theo thứ tự chủ đề trên |
| Bài test model gốc giải đúng | 194 |
| Bước / rollout mỗi nhánh | 128 / 512 |
| Giới hạn prompt / generation / full gold | 512 / 1024 / 1536 token |
| Learning rate / clip / policy epochs | 5e-6 / 0.2 / 2 |
| QLoRA | NF4 4-bit, double quantization; rank 8, alpha 16, dropout 0; `q_proj`, `v_proj` |

Hai nhánh train độc lập từ model gốc, cùng split và thứ tự bài. Dữ liệu được đóng băng trong `outputs/.../splits.json`; các tập không trùng UID hoặc nội dung đề chuẩn hóa. **Đây là subset MATH có đáp án nguyên, không phải MATH đầy đủ.** Verifier chưa hỗ trợ phân số, biểu thức hoặc tập hợp. Bài test bị cắt vẫn nằm trong metric chính.

## Kết quả

| Metric | Base model | GRPO | GRPO + ràng buộc |
|---|---:|---:|---:|
| Đúng / 640 test | 194 | 199 | 200 |
| Accuracy | 30,31% | 31,09% | 31,25% |
| Đúng → sai / 194 bài ban đầu đúng | — | 20 | 17 |
| Tỷ lệ đúng → sai | — | 10,31% | 8,76% |
| Sai → đúng | — | 25 | 23 |
| Thời gian train, gồm audit | — | 3,32 giờ | 3,89 giờ |

Cả pipeline hoàn tất trong **15,85 giờ**, dùng lại cache suy luận model gốc đã có. Thời gian này không bao gồm chi phí tạo cache ở lượt trước. Nhánh ràng buộc giữ được thêm **3 bài đúng**, nhưng sửa ít hơn **2 bài sai**, nên accuracy ròng hơn GRPO **1 bài**. Train tăng khoảng 17%, thêm 34 phút.

Đúng→sai theo chủ đề: đại số 9→7; hình học 2→1; lý thuyết số 4→4; tổ hợp/xác suất 5→5. Trên cùng 615 bài không bị cắt ở cả ba model, flip giảm 19→16. Đây là chẩn đoán bổ sung; subset này được chọn theo đầu ra, không thay thế toàn bộ test.

Chênh lệch ràng buộc trừ GRPO: accuracy **+0,156 điểm phần trăm**, bootstrap 95% **[-0,781; +1,094]**; flip rate **-1,546 điểm phần trăm**, khoảng **[-3,941; +0,532]**. Cả hai khoảng còn chứa 0. Bootstrap 2000 lần giữ cặp dự đoán giữa hai nhánh và lấy mẫu câu theo chủ đề. Một seed không đo được độ biến thiên giữa training seed.

**Kết luận:** tín hiệu giảm hồi quy đáp án nhỏ, accuracy gần tương đương, chi phí train tăng. Chưa đủ chứng minh tốt hơn ổn định. Test đã được xem trước khi quyết định chạy lại feedback cửa sổ, nên đây là đánh giá thăm dò hậu nghiệm, không phải test xác nhận độc lập.

Bản kết quả để đọc/chia sẻ: [comparison.csv](results/comparison.csv), [paired_statistics.csv](results/paired_statistics.csv), [run_timing.json](results/run_timing.json).

## Cấu trúc

```text
configs/math_retention.json     Cấu hình duy nhất của lượt được giữ
prepare_math_data.py            Kiểm tra dữ liệu đã đóng băng
train.py                       GRPO và GRPO + ràng buộc
src/plmco/math_trainer.py       Objective và vòng lặp train
src/plmco/retention.py          Chọn anchor và cập nhật lambda
src/plmco/modeling.py           Model, QLoRA và precision
src/plmco/math_model.py         Generation, verifier và gold loss
src/plmco/math_data.py          Schema, lọc và kiểm tra split
src/plmco/paired_statistics.py  Thống kê so sánh theo cặp
src/plmco/config.py, utils.py   Cấu hình và tiện ích
run_replication.py             Runner đầy đủ cho lượt này
evaluate.py                    Dự đoán validation/test
compare.py                     Metric và bootstrap
screen_report.py               Bảng kết quả trong terminal
visualize_results.ipynb        Chỉ plot lượt này
results/                       Bản CSV, timing và lịch lambda của lượt này
outputs/                       Dữ liệu, adapter, log và dự đoán chi tiết
hf_cache/                      Cache model/dataset phục vụ lượt này
tests/                         Kiểm tra objective, feedback, audit và metric
```

## Chạy trong VS Code

Chọn `C:\Users\slywi\anaconda3\envs\tf_gpu\python.exe`. Code không thay đổi CUDA hay TensorFlow. Config mặc định duy nhất là `math_retention.json`; bỏ biến `PLMCO_CONFIG` cũ nếu bạn đã đặt nó bên ngoài.

- **Chỉ xem kết quả:** mở `visualize_results.ipynb`, chọn Run All. Notebook chỉ còn plot hiện tại và giữ các output đã chạy. Nếu không có `outputs/` (ví dụ tải từ GitHub), notebook tự đọc CSV và lịch lambda đã lưu trong `results/`; không cần GPU để xem plot.
- **Tạo lại bảng từ dự đoán đã có:** chạy `compare.py`, rồi `screen_report.py`, rồi notebook. Không train hay generate lại.
- **Chạy toàn bộ quy trình:** chạy `run_replication.py`; hoặc lần lượt `prepare_math_data.py` → `train.py` → `evaluate.py` → `compare.py` → `screen_report.py` → notebook. Nhánh/đánh giá hoàn tất được bỏ qua.

Project không còn phụ thuộc output của lượt khác. Dữ liệu và cache cần thiết đã nằm trong lượt được giữ. Nếu xóa frozen split, code từ chối tự tạo một split khác dưới cùng tên run. Nếu thay cấu hình, phải tạo run mới và chuẩn bị dữ liệu mới có provenance phù hợp. Nhánh train bị ngắt không resume optimizer, mà khởi tạo lại.

Một số trường cấu hình lịch sử không dùng cho hai method hiện tại được giữ trong JSON/schema **chỉ để bảo toàn hash của dữ liệu đã đóng băng**. Không có nhánh CoKL/reference-KL hoặc feedback tức thì trong trainer hiện tại. Adapter/model cache/output chi tiết không được theo dõi bởi Git; cần giữ riêng nếu muốn tái lập lượt đã chạy. Repo có đủ bảng và lịch lambda để xem plot; để train lại đúng tập này cần frozen split, cache và model trên máy local, không thể chỉ clone repo rồi train mà thiếu các dữ liệu đó.
