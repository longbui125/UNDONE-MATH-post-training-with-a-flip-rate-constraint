# UNDONE — Constrained post-training for multi-topic math reasoning

Đây là **pilot chưa hoàn tất** cho đề tài *Post-training Language Models via Constrained Optimization*. Thí nghiệm kiểm tra liệu một ràng buộc mềm trong GRPO có giảm số bài toán mà model ban đầu giải đúng nhưng giải sai sau post-training hay không. Bốn chủ đề được thử là đại số, hình học, lý thuyết số, và tổ hợp/xác suất.

## Bài toán và phương pháp

Model gốc là Qwen2.5-Math-1.5B-Instruct. Cả ba nhánh huấn luyện cùng checkpoint, seed, dữ liệu, 64 nhóm bài và 4 lời giải lấy mẫu mỗi nhóm. Chỉ các adapter LoRA `q_proj` và `v_proj` được cập nhật (`rank=8`). Các lời giải được chấm đúng khi đáp án cuối là một số nguyên trong `\boxed{...}`.

| Nhánh | Cập nhật |
|---|---|
| `grpo` | GRPO với reward đúng/sai và advantage chuẩn hóa trong nhóm. |
| `replay_grpo` | GRPO cộng loss học lại lời giải mẫu của bài cũ, trọng số cố định 0,15. |
| `flip_constrained_grpo` | Replay GRPO cộng loss bảo vệ được điều chỉnh riêng cho từng chủ đề. |

Trước khi train, model gốc giải một tập **anchor** riêng; chỉ các bài nó giải đúng được dùng để theo dõi việc chuyển đúng → sai. Ở mỗi bước của nhánh ràng buộc, model giải lại một anchor. Sau mỗi 8 bước, tỷ lệ sai quan sát được của từng chủ đề điều chỉnh trọng số bảo vệ `λ` theo mục tiêu 10%, với `λ` nằm trong `[0, 1]`:

```text
loss = loss_GRPO + 0.15 × loss_replay + λ_topic × loss_retention
λ_topic ← clip(λ_topic + 0.5 × (observed_flip_rate − 0.10), 0, 1)
```

`loss_retention` là negative log-likelihood của lời giải mẫu trên anchor mà model gốc làm đúng. Đây là **surrogate constraint**: nó tăng hoặc giảm sức bảo vệ theo tín hiệu quan sát được, nhưng không bảo đảm tỷ lệ đúng → sai trên test thấp hơn 10%. Khi 4 lời giải trong nhóm đều có cùng reward, GRPO không tạo gradient; replay và retention vẫn có thể cập nhật adapter.

## Kết quả pilot

Một seed (42), 80 bài held-out test (20 bài/chủ đề), giới hạn sinh 384 token khi train **và** đánh giá. Kết quả chi tiết được lưu trong [results/pilot_seed42_comparison.csv](results/pilot_seed42_comparison.csv).

| Model | Đúng / 80 | Đúng → sai / 30 bài ban đầu đúng | Sai → đúng | Bị cắt / 80 |
|---|---:|---:|---:|---:|
| Gốc | 30 | — | — | 49 |
| GRPO | 30 | 2 | 2 | 49 |
| GRPO + replay | 30 | 2 | 2 | 48 |
| Replay + ràng buộc | 30 | 1 | 1 | 47 |

Ràng buộc được kích hoạt trong **12 bước cập nhật**. Trên test, số ca đúng → sai quan sát được giảm từ 2 xuống 1 so với hai nhánh không ràng buộc, nhưng số ca sai → đúng cũng giảm từ 2 xuống 1; tổng accuracy vẫn là **37,5%** ở mọi nhánh.

**Giới hạn diễn giải:** Tất cả ca đúng → sai ở ba nhánh huấn luyện đều là lời giải chạm trần 384 token trước khi có đáp án cuối. Chỉ 27/80 bài không bị cắt ở **cả bốn** model; trên tập chung này, model gốc và nhánh ràng buộc đúng 25 bài, còn GRPO và replay đúng 26 bài. Không nhánh nào có ca đúng → sai trong 27 bài đó. Vì vậy pilot **không chứng minh** ràng buộc chống quên cách giải toán. Nó chỉ cho thấy một ca chuyển đúng → sai ít hơn **dưới ngân sách 384 token**. Tập chung không bị cắt được chọn theo đầu ra của model và gần như đã giải đúng hết, nên chỉ dùng để chẩn đoán, không dùng thay kết quả toàn bộ test. Một seed và chênh lệch một bài cũng chưa đủ để xếp hạng phương pháp.

## Dữ liệu và cách chạy

Dữ liệu là tập con số nguyên của `HuggingFaceH4/MATH`, cố định ở revision `9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9`. MATH train được tách thành train, anchor và validation; MATH test là held-out test. Chỉ giữ bài có đáp án mẫu là số nguyên trong `\boxed{...}` và lời giải mẫu đủ ngắn cho cấu hình. Các tập không trùng câu hỏi.

Trong VS Code, chọn môi trường Python có các gói trong [requirements.txt](requirements.txt). Sửa `model_name` trong [configs/deadline_pilot.json](configs/deadline_pilot.json) thành đường dẫn model trên máy hoặc Hugging Face model ID, rồi chạy một trong hai cách:

1. Chạy [run_deadline_pilot.py](run_deadline_pilot.py) để thực hiện tuần tự toàn bộ pilot; hoặc
2. Chạy lần lượt [prepare_math_data.py](prepare_math_data.py), [train.py](train.py), [evaluate.py](evaluate.py), [compare.py](compare.py), [screen_report.py](screen_report.py) với biến môi trường `PLMCO_CONFIG=deadline_pilot.json`.

Sau đó mở [visualize_results.ipynb](visualize_results.ipynb) và chạy các cell. Notebook giữ biểu đồ trên **toàn bộ test**, đồng thời có phần riêng trên **cùng 27 bài không bị cắt** để kiểm tra tác động của giới hạn token. Bảng số liệu và đường biến thiên trọng số bảo vệ được hiển thị kèm theo.

`configs/pilot.json` là cấu hình 320 bước để mở rộng về sau; kết quả của cấu hình này **chưa được báo cáo** ở đây. Không được ghép checkpoint hay metric giữa cấu hình 64 bước và 320 bước. `outputs/` và `hf_cache/` được bỏ qua khi commit vì chứa dữ liệu sinh, adapter và model cache; chúng sẽ được tạo lại tại máy chạy.
