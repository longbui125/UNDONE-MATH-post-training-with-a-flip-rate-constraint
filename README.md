# Hậu huấn luyện GRPO với ràng buộc tỷ lệ đúng thành sai

Dành cho trình bày: [báo cáo chi tiết (PDF)](reports/bao_cao_grpo_rang_buoc_flip.pdf) và [bản nguồn có thể chỉnh sửa](reports/bao_cao_grpo_rang_buoc_flip.md).

**Cập nhật kiểm tra seed 43 (02/10/2026):** phát hiện loader đánh giá v1 và loader train QLoRA khác precision. Một anchor đã đổi đáp án ngay khi chuẩn bị QLoRA, **trước bất kỳ optimizer update nào**. Vì vậy phần “flip trên anchor do training” của v1 bị nhiễu bởi loader; các bảng cũ bên dưới được giữ như kết quả lịch sử, chưa chứng minh giảm quên thuần túy. Xem [chẩn đoán](reports/seed43_diagnosis.md). Lượt v2 chạy lại cả GRPO và ràng buộc trên seed 43/45 với loader đồng nhất, không dùng lại baseline hay adapter v1.

Dự án hỏi liệu một mô hình ngôn ngữ tổng quát có thể học thêm toán bằng GRPO mà ít làm sai những bài trước đó nó giải đúng hơn hay không. Thử nghiệm dùng [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Seed 42 đánh giá bốn nhánh; seed 43 và 44 lặp lại cặp GRPO và GRPO + ràng buộc. Đây vẫn là pilot, chưa đủ để kết luận phương pháp tốt hơn một cách ổn định.

## Baseline đã chốt

Mọi nhánh bắt đầu từ cùng checkpoint và dùng cùng train/validation/test, thứ tự bài, seed, group size, số bước và giới hạn sinh.

| Nhánh | Thuật toán |
|---|---|
| `initial` | Model ban đầu, không train. |
| `grpo` | GRPO không regularizer, theo [DeepSeekMath](https://arxiv.org/abs/2402.03300). |
| `grpo_reference_kl` | GRPO cộng reverse-KL ước lượng trên rollout tới model ban đầu đóng băng; đây là đối chứng KL phổ biến trong GRPO. |
| `cokl_grpo` | GRPO cộng CoKL theo [Wang et al.](https://arxiv.org/html/2608.01743v1): hai vế reference-correct và current-correct cùng hiệu chỉnh chuẩn hóa theo công thức Eq. 17. |
| `flip_constrained_grpo` | Phương pháp đang khảo sát: loss bảo vệ bài ban đầu đúng có trọng số thích nghi theo tỷ lệ đúng → sai trên anchor. |

CoKL **cần** một buffer đáp án đúng do model gốc sinh sẵn, như paper mô tả. Buffer này không phải nhánh `GRPO + replay` độc lập, nhưng vẫn là dữ liệu tham chiếu được tái sử dụng; không được mô tả CoKL là hoàn toàn không dùng dữ liệu cũ. Cấu hình local dùng 2 đáp án tham chiếu trên một anchor và 4 đáp án hiện tại trên một bước. Đây là triển khai đối chứng theo công thức CoKL dưới hạn chế compute, **không tái hiện nguyên quy mô/thiết lập của paper**. Tiền xử lý buffer được tính riêng vào chi phí.

Phương pháp ràng buộc đánh giá một anchor ban đầu đúng mỗi bước. Sau cửa sổ 32 bước, hệ số bảo vệ riêng từng chủ đề được cập nhật theo `λ ← clip(λ + 0.30 × (flip_rate − 0.10), 0, 1)`. Loss train là GRPO loss cộng `λ × gold-solution loss` của anchor. Đây là ràng buộc **mềm trên anchor**, không bảo đảm cứng rằng test sẽ không có flip. Model, anchor và verifier giống nhau giữa các nhánh; mỗi paper method được ghi đúng cơ chế và phần điều chỉnh local của nó.

## Dữ liệu, ngân sách và giới hạn kết luận

[Config mặc định](configs/general_math_baseline.json) dùng 4 chủ đề từ revision cố định của `HuggingFaceH4/MATH`: đại số, hình học, lý thuyết số, tổ hợp và xác suất. Mỗi chủ đề có 16 train, 16 anchor, 20 validation và 40 held-out test: tổng **64/64/80/160** bài. Config gốc đóng băng seed 42; [run_replication.py](run_replication.py) dùng thêm seed 43 và 44 cho hai nhánh chính. Mỗi nhánh có **64 bước**, 4 rollout/group, 2 policy epochs. Train, đánh giá và kiểm tra anchor đều cho sinh tối đa **1024 token**, cao hơn 2,67 lần mức 384 token của pilot trước. Gold anchor dùng giới hạn tổng prompt + lời giải là 1536 token. Tất cả nhánh dùng cùng trần 1024; tỷ lệ chạm trần được lưu và báo cáo.

Paper CoKL dùng trần sinh 8192 token trong thực nghiệm chính; **1024 token không phải cấu hình gốc của paper**. Đây là ngân sách local trên RTX 3060 6 GB. Lượt chạy seed 42 hoàn tất trong **18 giờ 29 phút** tính cả chuẩn bị, train và đánh giá; riêng train bốn nhánh khoảng **12 giờ 16 phút**. Lượt lặp seed 43–44 mất thêm **11 giờ 53 phút** tính cả train và đánh giá. Ba seed và 64 bước mỗi nhánh chỉ đủ cho **baseline thăm dò**, chưa đủ để khẳng định cải thiện có ý nghĩa thống kê.

**Đây là subset MATH có đáp án nguyên, không phải benchmark MATH đầy đủ.** Verifier hiện xác minh số nguyên cuối trong `\boxed{...}`; phân số, biểu thức và tập hợp chưa được hỗ trợ. Split được đóng băng và kiểm tra trùng đề. `selection_audit` trong `splits.json` lưu số câu bị loại vì kiểu đáp án, trùng hoặc prompt quá dài. Độ dài lời giải mẫu chỉ lọc tập anchor cần gold loss; train, validation và test không bị lọc theo độ dài lời giải mẫu. Bài test bị chạm trần vẫn nằm trong metric chính; notebook có phân tích subset không bị chạm trần riêng, chỉ để chẩn đoán.

Ba nhánh GRPO dùng cùng RL objective cơ bản. `grpo_reference_kl` dùng hệ số 0.04; CoKL dùng hệ số 0.001 và clipped sequence importance ratio 0.2. Đây là giá trị khởi tạo cho thử nghiệm, **chưa được chọn bằng validation sweep**. Vì vậy không nên diễn giải kết quả như so sánh tối ưu hyperparameter của các paper. Code dùng QLoRA 4-bit trên `q_proj` và `v_proj`, cũng không tái hiện hạ tầng đầy đủ của paper. CoKL tốn thêm generation; so sánh cả accuracy và thời gian/token.

## Kết quả seed 42

| Nhánh | Đúng / 160 test | Đúng → sai / 47 bài ban đầu đúng | Sai → đúng | Thời gian train |
|---|---:|---:|---:|---:|
| Model gốc | 47 (29,38%) | 0 | 0 | — |
| GRPO | 49 (30,63%) | 10 | 12 | 1 giờ 15 phút |
| GRPO + reference-KL | 43 (26,88%) | 11 | 7 | 1 giờ 22 phút |
| CoKL-GRPO | 45 (28,13%) | 10 | 8 | 7 giờ 14 phút |
| GRPO + ràng buộc | 50 (31,25%) | 7 | 10 | 2 giờ 25 phút |

So với GRPO, ràng buộc làm sai ít hơn 3 bài vốn đúng nhưng cũng sửa đúng ít hơn 2 bài vốn sai; chênh lệch accuracy ròng là **1/160 bài**. Trên cùng 151 bài test không bị chạm trần ở bất kỳ nhánh nào, GRPO đúng 47 bài và có 9 ca đúng → sai; nhánh ràng buộc đúng 50 bài và có 6 ca đúng → sai. Phân tích này chỉ để kiểm tra ảnh hưởng của cắt token vì tập 151 bài được chọn theo đầu ra. Tỷ lệ đúng → sai của nhánh ràng buộc trên toàn test vẫn là **7/47 = 14,9%**, cao hơn mục tiêu 10% đặt cho anchor; ràng buộc mềm không được xem là đã đạt mục tiêu cứng. [Bảng seed 42](results/seed42_summary.csv) và [notebook có biểu đồ](visualize_results.ipynb) lưu kết quả để xem lại. Adapter, dự đoán chi tiết và model cache nằm trong `outputs/` hoặc `hf_cache/` trên máy chạy, không đưa lên Git.

## Kết quả lặp lại trên seed 42–44

Hai nhánh chính dùng cùng model gốc (47/160 đúng), cùng split và cùng trần sinh. Mỗi seed train lại **độc lập** từ checkpoint gốc, không train nối tiếp. Bảng dưới là accuracy trên toàn bộ 160 bài test và số bài trong 47 bài model gốc giải đúng bị chuyển thành sai:

| Seed | GRPO đúng /160 | Ràng buộc đúng /160 | GRPO đúng → sai | Ràng buộc đúng → sai | GRPO sai → đúng | Ràng buộc sai → đúng |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 49 | 50 | 10 | 7 | 12 | 10 |
| 43 | 44 | 45 | 8 | 10 | 5 | 8 |
| 44 | 45 | 50 | 11 | 9 | 9 | 12 |

Nhánh ràng buộc hơn GRPO về accuracy ở cả ba seed, trung bình **30,21% so với 28,75%**. Nhưng khả năng giảm đúng → sai **không ổn định**: seed 43 có 10 ca flip, nhiều hơn GRPO 2 ca. Thời gian train cộng ba seed của nhánh ràng buộc khoảng **1,79 lần** GRPO. Các seed dùng lại cùng 160 câu test, nên không diễn giải tổng số lượt đánh giá như các câu hỏi độc lập. [CSV chi tiết](results/seed_replication_comparison.csv) giữ cả seed 43; [notebook](visualize_results.ipynb) đặt biểu đồ đủ ba seed làm kết quả chính. Biểu đồ gộp riêng 42+44 chỉ là góc nhìn **thăm dò được chọn sau khi đã xem seed 43**, không dùng để khẳng định phương pháp giảm flip ổn định.

## Chạy trong VS Code

### Lượt đang chuẩn bị: chạy lại seed 43 và thêm seed 45

Chọn Python `C:\Users\slywi\anaconda3\envs\tf_gpu\python.exe`, rồi:

1. Chạy **[run_replication.py](run_replication.py)** bằng Run Python File. File tự chọn [general_math_feedback_v2.json](configs/general_math_feedback_v2.json), sao chép đúng các câu đã đóng băng, quét lại anchor, train hai nhánh ở seed 43, đánh giá và lập bảng; sau đó làm tương tự seed 45. Không cần chạy lại các file prepare/train/evaluate/compare riêng cho lượt này.
2. Mở **[visualize_results.ipynb](visualize_results.ipynb)** và Run All. Phần cuối **Feedback v2 — seeds 43 and 45** hiển thị accuracy, đúng → sai, sai → đúng, thời gian, từng chủ đề và trọng số thực sự được dùng. Khi mới xong seed 43, phần này vẫn xem được kết quả tạm thời của seed đó.

Kết quả mới nằm trong `outputs/general_math_feedback_1024_v2/`: `seed_43/`, `seed_45/`, `comparison.csv`, `retention_baseline.json`, `seed_replication_plan.json` và `seed_replication_timing.json`. Kết quả v1 ở folder cũ được giữ nguyên. Chạy lại file sẽ bỏ qua nhánh hoàn thành; nhánh dở phải chạy lại từ đầu, **không resume optimizer**. Hai nhánh và hai seed đều khởi tạo độc lập từ model gốc, không nối tiếp adapter seed 43 sang 45.

Các thay đổi có chủ đích của v2:

- **Đồng nhất precision:** scan anchor, initial và đánh giá adapter đều dùng bước chuẩn bị k-bit giống train (các tham số không quantize được đưa lên FP32). Quét lại base-correct anchor và đánh giá lại initial; không dùng 47 bài initial-correct hay 17 anchor của v1 làm mẫu số mới.
- **Kiểm tra trước train ở cả hai nhánh:** zero-initialized LoRA phải giải đúng các anchor đã chọn. Nếu không, dừng và ghi `initial_anchor_audit.json` thay vì tính chênh lệch loader là quên kiến thức. Token và thời gian audit được lưu riêng; thời gian summary đã bao gồm audit.
- **Phản hồi ngay trong bước hiện tại:** sau mỗi lần kiểm tra anchor, cập nhật hệ số của chủ đề đó **trước** khi backward. Không còn 32 bước đầu bỏ mặc các tín hiệu flip, hoặc cập nhật ở bước cuối rồi không dùng.
- **Đếm bài khác nhau:** dùng trạng thái kiểm tra gần nhất của tối đa 4 UID anchor khác nhau/chủ đề; kiểm tra lại một UID sẽ thay thế trạng thái cũ, không nhân số phiếu của bài đó. Đây là ước lượng từ các lần kiểm tra không đồng thời, chưa phải tỷ lệ của toàn bộ anchor ở checkpoint hiện tại. Nếu chủ đề chỉ có 1–2 anchor đúng, độ phủ vẫn ít.

Công thức mới vẫn là `λ_topic ← clip(λ_topic + 0.30 × (recent_flip_rate_topic − 0.10), 0, 1)` và `loss = GRPO_loss + λ_topic × gold_solution_loss`. Loss gold là negative mean log-probability trên token lời giải mẫu, không tính token đề bài. Chỉ thời điểm cập nhật và cách ước lượng feedback được sửa; GRPO objective, reward, dataset, learning rate, 64 bước, 4 rollout/group, 2 policy epochs và trần 1024 token được giữ. Đây vẫn là ràng buộc mềm, không có bảo đảm mọi bài test sẽ không flip.

Log mới ghi `retention_weight_used`, `retention_feedback_wrong/total`, `retention_gold_loss` (epoch cuối), `optimizer_updates`. Chi phí anchor generation và audit được báo cáo thêm, không chỉ rollout tokens. Hai nhánh có cùng ngân sách lấy mẫu RL; số optimizer update có thể khác vì nhánh ràng buộc vẫn có anchor loss khi group không có advantage. Đây là khác biệt cơ chế cần báo cáo, không phải so sánh bằng cùng mọi chi phí.

Không sửa phương pháp bằng cách chọn seed cho kết quả đẹp: seed 43 cũ vẫn được giữ, seed 43 mới là kiểm tra sau sửa, seed 45 là seed bổ sung đã chọn trước khi có kết quả. Chúng dùng lại cùng test đã xem; không gọi là test hoàn toàn mới. Chỉ kết luận có tiến triển nếu flip giảm mà accuracy/gains không bị chặn quá nhiều, và tính cả thời gian/token tăng thêm. **Chưa chạy train v2 và chưa khẳng định seed 43 đã tốt hơn.**

### Quy trình v1 đã hoàn thành (lưu để tái lập)

Chọn Python của môi trường `tf_gpu`. Không sửa CUDA hay TensorFlow. Cài dependency còn thiếu từ [requirements.txt](requirements.txt) nếu cần. Model tổng quát phải tải được từ Hugging Face hoặc đã có local. Chạy lần lượt:

1. [prepare_math_data.py](prepare_math_data.py) — tạo split bất biến.
2. [train.py](train.py) — quét anchor, chuẩn bị CoKL reference buffer, rồi train bốn nhánh.
3. [evaluate.py](evaluate.py) — đánh giá validation và held-out test khi tất cả nhánh đã train xong.
4. [compare.py](compare.py) — tạo `outputs/general_math_paper_baselines_1024_v1/comparison.csv`.
5. [screen_report.py](screen_report.py) — bảng tóm tắt.
6. [visualize_results.ipynb](visualize_results.ipynb) — plot so sánh.

Hoặc chạy [run_experiment.py](run_experiment.py) cho toàn bộ theo thứ tự. Nếu cần thay model hoặc ngân sách, tạo `run_name` mới trước khi chuẩn bị split.

### Kiểm chứng thêm seed 43 và 44

Lượt lặp v1 đã chạy tuần tự GRPO và GRPO + ràng buộc ở seed 43 và 44, rồi đánh giá, so sánh và in bảng. Kết quả nằm cạnh run gốc trong `outputs/general_math_paper_baselines_1024_v1/seed_43`, `seed_44`, `seed_replication_comparison.csv` và `seed_replication_timing.json`. Notebook giữ biểu đồ ba seed và góc nhìn thăm dò 42+44. **`run_replication.py` hiện đã chuyển sang v2 seed 43/45**, không chạy lại v1 khi bấm Run.

Hai seed mới đều bắt đầu từ **cùng model gốc**, cùng các tập đã đóng băng (64 train, 64 anchor, 80 validation, 160 test), cùng hyperparameter và 64 bước; trong mỗi seed, hai phương pháp dùng cùng thứ tự bài. Mỗi chủ đề có 16 bài train khác nhau; thứ tự bài và lượt lấy mẫu thay đổi theo seed. `seed_replication_plan.json` lưu lịch bài chính xác và kiểm tra các seed thực sự có thứ tự khác nhau. Log mới ghi thêm số lời giải khác nhau trong mỗi group, cùng `mixed_group_rate` để đánh giá độ đa dạng khi chạy. Model gốc và tập anchor đúng ban đầu được dùng lại; CoKL không chạy trong lượt kiểm chứng hai phương pháp chính. Các nhánh hoàn thành được bỏ qua khi chạy lại file sau gián đoạn; nhánh dở sẽ chạy lại từ đầu.

Trong dữ liệu đã đóng băng, thứ tự câu train khác seed 42 ở **53/64 vị trí** cho seed 43 và **58/64 vị trí** cho seed 44; hai seed mới khác nhau ở **61/64 vị trí**. Tập anchor model gốc giải đúng chỉ có 7 đại số, 2 hình học, 4 lý thuyết số và 4 tổ hợp/xác suất; đây vẫn là hạn chế của phép đo ràng buộc theo chủ đề, không thể được sửa bằng cách chỉ đổi seed.

Lượt chạy seed 43–44 đã hoàn tất trong **11,89 giờ** tính cả train và đánh giá. Đây là kiểm tra lặp lại trên ba seed và cùng một test đã xem ở seed 42; không diễn giải nó như một bộ test hoàn toàn mới hoặc bằng chứng phương pháp đã vượt mọi đối chứng.

## Cách đọc kết quả

Chỉ số chính là **accuracy trên toàn bộ test**, **số và tỷ lệ đúng → sai trên những câu model gốc giải đúng**, và **số sai → đúng**. Xem thêm từng chủ đề, mức yếu nhất, tỷ lệ chạm trần, tỷ lệ không có `\boxed{...}`, mixed-group rate, số optimizer update, rollout tokens, CoKL reference/current tokens, anchor-check tokens và thời gian. Muốn nói ràng buộc có lợi thì cần cho thấy nó giảm flip mà không chặn phần lớn tiến bộ accuracy. Ba seed hiện cho tín hiệu accuracy nhưng chưa cho thấy giảm flip ổn định; hai đối chứng có paper chỉ được chạy ở seed 42.
