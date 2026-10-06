# GRPO với ràng buộc mềm để giảm tỷ lệ đúng → sai

**Đề tài:** Post-training Language Models via Constrained Optimization.

Repo ghi lại một thử nghiệm thăm dò với **Qwen2.5-1.5B-Instruct**, so sánh **GRPO** và **GRPO + ràng buộc theo chủ đề** (`flip_constrained_grpo`). Lượt được giữ: **seed 42, 128 bước, 640 bài test**, hoàn tất ngày 05/10/2026. Tài liệu cập nhật ngày 06/10/2026.

**Kết quả cần nhớ:** đúng → sai giảm từ **20 xuống 17 bài**, nhưng sai → đúng cũng giảm từ **25 xuống 23 bài**. Số bài đúng cuối cùng là **199/640 so với 200/640**. Đây là tín hiệu nhỏ, **chưa đủ khẳng định phương pháp tốt hơn ổn định**.

- [Báo cáo PDF chi tiết, dễ hiểu](output/pdf/bao_cao_grpo_rang_buoc_seed42.pdf)
- [Notebook biểu đồ và kết quả](visualize_results.ipynb)
- [Bảng kết quả](results/comparison.csv) và [thống kê theo cặp](results/paired_statistics.csv)

## 1. Bài toán: học thêm nhưng có thể làm sai bài từng đúng

Một model ban đầu giải đúng một số bài, sai một số bài khác. Sau hậu huấn luyện, có bốn trường hợp trên cùng câu hỏi:

| Trước train | Sau train | Ý nghĩa |
|---|---|---|
| Đúng | Đúng | Giữ được đáp án đúng |
| Đúng | Sai | Hồi quy đáp án: **đúng → sai** |
| Sai | Đúng | Cải thiện: **sai → đúng** |
| Sai | Sai | Chưa cải thiện |

Chỉ nhìn accuracy tổng có thể bỏ qua hiện tượng này. Sửa được 10 bài sai nhưng làm sai 10 bài đúng thì accuracy không đổi, dù hành vi trên từng bài thay đổi.

**Mục tiêu:** giảm đúng → sai bằng một ràng buộc mềm trong quá trình GRPO, đồng thời đo xem model còn sửa được bao nhiêu bài sai và phải trả thêm bao nhiêu chi phí.

Một khả năng giải thích là các bài dùng chung tham số: cập nhật có lợi cho bài này có thể bất lợi cho bài khác. Tuy nhiên, run này **chưa chứng minh nguyên nhân bằng phân tích gradient**. Đúng → sai là hồi quy đầu ra dưới cấu hình đánh giá hiện tại; không tự động chứng minh model mất hoàn toàn một kỹ năng.

## 2. Model, dữ liệu và thiết lập

### Model ban đầu và model đang train

Model là **`Qwen/Qwen2.5-1.5B-Instruct`**, đã được tiền huấn luyện và instruction tuning. Lượt này không huấn luyện từ đầu, không dùng Qwen-Math và **không có giai đoạn SFT riêng trước RL**.

QLoRA giữ trọng số gốc cố định, chỉ cập nhật adapter:

```math
W_{\text{hiệu dụng}}=W_{\text{gốc}}+\Delta W_{\text{LoRA}}.
```

“Model ban đầu” và “model hiện tại” dùng cùng kiến trúc và trọng số gốc. Adapter thay đổi qua các bước train, làm xác suất sinh lời giải thay đổi. **Loss anchor được tính bằng model với adapter hiện tại**, không phải lấy loss từ một base model cố định rồi dùng nó để train.

### Dữ liệu và vai trò từng tập

Nguồn: `HuggingFaceH4/MATH`, revision `9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9`.

Chọn bài có đáp án nguyên trong boxed answer cuối và prompt vừa giới hạn. Anchor còn phải có đề + lời giải chuẩn + EOS vừa giới hạn 1536 token. **Đây là subset MATH, không phải MATH đầy đủ.** Verifier chưa hỗ trợ phân số, biểu thức hoặc tập hợp.

| Tập | Tổng số | Mỗi chủ đề | Vai trò |
|---|---:|---:|---|
| Train | 128 | 32 | Sinh lời giải và tính loss GRPO |
| Anchor ứng viên | 512 | 128 | Chọn bài model ban đầu giải đúng để theo dõi/giữ khả năng |
| Validation | 160 | 40 | Đánh giá bổ sung; không dùng làm anchor |
| Test | 640 | 160 | So sánh kết quả cuối trên cùng câu hỏi |

Bốn chủ đề: **đại số, hình học, lý thuyết số, tổ hợp và xác suất**. Các tập được đóng băng; kiểm tra không trùng UID hoặc nội dung đề chuẩn hóa giữa các tập.

Model ban đầu giải đúng **153 anchor** và **194 bài test**. Đây là hai tập khác nhau:

| Chủ đề | Anchor được chọn | Bài test ban đầu đúng |
|---|---:|---:|
| Đại số | 64 | 83 |
| Hình học | 22 | 34 |
| Lý thuyết số | 38 | 35 |
| Tổ hợp và xác suất | 29 | 42 |
| **Tổng** | **153** | **194** |

Không dùng test để train hoặc cập nhật λ. Tuy nhiên, test đã được xem khi chọn hướng thử nghiệm; đây là hạn chế của đánh giá thăm dò, nêu ở phần 9.

### Đa chủ đề được train như thế nào?

Bài được xáo trộn trong từng chủ đề rồi chọn luân phiên: đại số → hình học → lý thuyết số → tổ hợp/xác suất → lặp lại. Không train xong toàn bộ chủ đề này rồi mới chuyển sang chủ đề khác. Bài trong tập train không nhất thiết là bài model chưa từng biết hoặc đang giải sai.

| Tham số | Thiết lập |
|---|---|
| Training seed / data seed | 42 / 20261002 |
| Bước / group size / policy epochs | 128 / 4 / 2 |
| Optimizer / learning rate | AdamW / 5e-6 |
| Ratio clip / gradient clip | 0,2 / 1,0 |
| Sampling train | Temperature 0,8; top_p 0,95 |
| Đánh giá và kiểm tra anchor | Greedy decoding |
| Prompt / lượt sinh / đề + gold solution | 512 / 1024 / 1536 token |
| QLoRA | NF4 4-bit, double quantization; rank 8, alpha 16, dropout 0; q_proj, v_proj |
| Feedback | Cửa sổ 32 bước; ngưỡng 0,10; tốc độ điều chỉnh 0,30; λ trong [0; 1] |

## 3. Loss GRPO: học từ lời giải tự sinh

Với một đề, model sinh 4 lời giải. Bộ chấm đọc đáp án nguyên trong `\boxed{...}` cuối: đúng nhận reward 1; sai hoặc không đọc được đáp án theo định dạng nhận 0.

Ví dụ minh họa, không phải log thực nghiệm: “Chọn 3 người từ 6 người, có bao nhiêu cách?” Đáp án là 20.

| Lời giải | Đáp án sinh ra | Reward | Advantage |
|---|---:|---:|---:|
| 1 | 20 | 1 | +1 |
| 2 | 15 | 0 | -1 |
| 3 | 20 | 1 | +1 |
| 4 | 10 | 0 | -1 |

Advantage đo mức tốt/kém so với trung bình nhóm:

```math
A_{i}=\frac{R_{i}-\overline{R}}{\sigma_{R}+10^{-8}}.
```

Ở ví dụ này trung bình và độ lệch chuẩn đều là 0,5. Model được khuyến khích tăng xác suất lời giải có advantage dương và giảm xác suất lời giải có advantage âm. Nhóm đều đúng hoặc đều sai không có tín hiệu phân biệt; code bỏ cập nhật GRPO của nhóm đó.

Với token j trong lời giải i, tỷ lệ xác suất là:

```math
\rho_{i,j}=\frac{p_{\theta}(y_{i,j}\mid x,y_{i,1:j-1})}{p_{\mathrm{old}}(y_{i,j}\mid x,y_{i,1:j-1})}.
```

Ký hiệu `y_{i,1:j-1}` là các token đứng trước token j trong lời giải i; với token đầu tiên, phần này rỗng. `p_old` được lưu trước các policy epoch của group hiện tại, **không phải luôn là base model trước toàn lượt train**.

```math
L_{\mathrm{GRPO}}=-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{i}}\sum_{j=1}^{T_{i}}
\min\left(\rho_{i,j}A_{i},\mathrm{clip}(\rho_{i,j},0.8,1.2)A_{i}\right).
```

G là số lời giải; T_i là số token của lời giải i. Code mean trên token từng lời giải rồi mean trên group. Dấu âm chuyển objective cần tối đa hóa thành loss cần tối thiểu hóa. Clipping hạn chế động lực thay đổi xác suất quá mạnh, không bảo đảm cứng mọi cập nhật đều nằm trong một khoảng xác suất.

Đây là comparator GRPO cục bộ với objective trên. **Hai nhánh không có reference-KL trong loss**, nên không claim tái lập toàn bộ setup paper GRPO gốc. Các trường reference-KL/CoKL lịch sử trong config không được trainer hiện tại sử dụng.

## 4. Loss anchor và cơ chế ràng buộc

### 4.1. Anchor được dùng như thế nào?

Mỗi bước lấy một anchor theo vòng luân phiên chủ đề, cho model hiện tại giải bằng greedy decoding rồi kiểm tra đáp án. Anchor đã đúng ban đầu, nên trả lời sai lúc này được ghi nhận là một lượt hồi quy. Đây là kiểm tra mẫu, **không đánh giá toàn bộ 153 anchor mỗi bước**; một số anchor có thể được chọn lại.

Ví dụ minh họa: `2x + 3 = 11` được model ban đầu giải đúng `x = 4`, nên có thể làm anchor. Trong train, nếu trả lời `x = 5`, ghi nhận một lượt đúng → sai. Lời giải chuẩn vẫn là trừ 3 để được `2x = 8`, rồi chia 2 để được `x = 4`.

### 4.2. Loss anchor là gì?

Đưa đề và **lời giải chuẩn từ dataset** vào model với adapter hiện tại. Dùng các token chuẩn trước đó để đo xác suất token chuẩn tiếp theo:

```math
L_{\mathrm{anchor}}=-\frac{1}{T}\sum_{j=1}^{T}\log p_{\theta}\left(y_{j}^{\ast}\mid x,y_{1:j-1}^{\ast}\right).
```

- Tính trên toàn bộ token lời giải chuẩn và EOS của **một bài anchor được chọn**; không tính token đề.
- Không tính loss trên toàn bộ tập anchor mỗi bước.
- Nhãn là lời giải dataset, không phải lời giải mà base model từng tự sinh.
- Đây là thành phần học có giám sát phụ trợ trong RL, dù không có bước SFT riêng trước RL.

Xác suất token chuẩn càng cao, loss càng nhỏ. Ví dụ: `p = 0,8` cho `-ln(p) ≈ 0,223`; `p = 0,2` cho loss khoảng 1,609. Giảm loss giúp model dự đoán lời giải chuẩn dễ hơn, không đồng nghĩa chắc chắn greedy decoding sẽ trả lời đúng.

### 4.3. λ tác động vào đâu?

```math
L_{\text{tổng}}=L_{\mathrm{GRPO}}+\lambda_{k}L_{\mathrm{anchor}}.
```

```math
\nabla L_{\text{tổng}}=\nabla L_{\mathrm{GRPO}}+\lambda_{k}\nabla L_{\mathrm{anchor}}.
```

λ_k là hệ số riêng cho chủ đề k của anchor. λ nhân sức tác động của gradient anchor; không sửa reward GRPO. λ = 0 thì chỉ kiểm tra anchor, chưa thêm loss; λ tăng thì thành phần giữ lời giải chuẩn tác động mạnh hơn; λ giảm thì tác động yếu đi.

**λ = 0,1 không có nghĩa “10% học anchor, 90% học GRPO”**, vì độ lớn hai gradient có thể khác nhau. Code tính gradient hai loss riêng rồi cộng dồn trước cùng một optimizer update. Không phải một bước chỉ GRPO rồi một bước chỉ anchor.

### 4.4. Điều chỉnh λ bằng tỷ lệ đúng → sai

λ khởi tạo 0. Sau mỗi cửa sổ **32 bước toàn cục**, mỗi chủ đề có 8 lượt kiểm tra trong cấu hình này:

```math
\widehat{f}_{k}=\frac{\text{số lượt anchor trả lời sai}}{\text{tổng lượt kiểm tra anchor}}.
```

```math
\lambda_{k}\leftarrow\mathrm{clip}\left[\lambda_{k}+0.30(\widehat{f}_{k}-0.10),0,1\right].
```

`0,10` là ngưỡng mục tiêu; `0,30` là tốc độ điều chỉnh λ, khác learning rate model; clip giữ λ trong [0; 1]. Vượt ngưỡng thì tăng λ; dưới ngưỡng thì giảm, không xuống dưới 0.

Ví dụ 1 lượt sai / 8 lượt kiểm tra = 12,5%. Nếu λ đang bằng 0, giá trị mới là `0 + 0,30 × (0,125 - 0,10) = 0,0075`.

**λ mới dùng từ cửa sổ tiếp theo.** 32 bước đầu chưa thêm anchor loss nhưng vẫn kiểm tra anchor. Cập nhật sau bước 128 được ghi log nhưng không còn bước train để dùng.

### 4.5. Vì sao gọi là ràng buộc mềm?

Tỷ lệ đúng → sai từ đáp án rời rạc không được lấy gradient trực tiếp. Nó điều khiển λ; **loss lời giải chuẩn là đại lượng thay thế có thể lấy gradient**.

Đây là ràng buộc mềm có phản hồi, **không phải projection, CPO hay lời giải chính xác của tối ưu với ràng buộc flip-rate**. Không bảo đảm mọi bài từng đúng đều đúng hoặc tỷ lệ test dưới 10%.

Khi λ > 0, anchor loss vẫn tạo update dù group RL không có advantage. Vì vậy cùng số bước/group RL không đồng nghĩa cùng số optimizer update hoặc compute.

## 5. Workflow và điều kiện so sánh

```text
Đóng băng train / anchor / validation / test
    ↓
Đánh giá ban đầu; chọn 153 anchor đã giải đúng
    ↓
Audit cả hai model có adapter mới trước train
    ↓
Khởi tạo hai adapter độc lập từ cùng model gốc
    ├─ GRPO: 4 lời giải → reward → advantage → loss GRPO → update
    └─ Ràng buộc: kiểm tra anchor + sinh 4 lời giải
                  → loss GRPO + λ × loss anchor → update
                  → sau 32 bước: điều chỉnh λ từng chủ đề
    ↓
Đánh giá initial và adapter cuối trên cùng validation/test
    ↓
So sánh từng câu → accuracy / flip / gain / chi phí / bootstrap / plot
```

Model train và đánh giá dùng cùng cách chuẩn bị QLoRA để tránh lệch precision. Audit trước train kiểm tra 153 anchor, hai nhánh đều không có mismatch. Seed được đặt lại sau audit để kiểm tra không làm lệch dòng ngẫu nhiên lấy mẫu RL.

Cùng model, split, seed, thứ tự bài, token limit và số group. Sau cập nhật, model sinh lời giải khác nhau; **không claim rollout giống hệt hoặc compute bằng nhau**. Chạy đủ 128 bước rồi đánh giá adapter cuối, không chọn checkpoint theo test.

## 6. Đọc metric

| Metric | Ý nghĩa |
|---|---|
| Accuracy | Số câu test đúng / tổng câu test |
| Macro accuracy | Trung bình bốn chủ đề; mỗi chủ đề 160 câu nên bằng accuracy toàn test trong run này |
| Worst topic | Accuracy thấp nhất trong bốn chủ đề |
| Correct → wrong | Câu model ban đầu đúng nhưng sau train sai |
| Flip-rate test | Correct → wrong / **194 câu test ban đầu đúng**, không chia cho 640 |
| Wrong → correct | Câu ban đầu sai nhưng sau train đúng |
| Mixed-group rate | Tỷ lệ group có cả reward 0 và 1 |
| Capped | Chạm trần token mà chưa sinh EOS; không mặc định đồng nghĩa đáp án sai |
| No box | Không có boxed answer hoàn chỉnh; không tự động chứng minh suy luận toán sai |
| Retention checks | Lượt kiểm tra anchor, không phải số anchor khác nhau |
| Retention active updates | Policy epoch có thêm gradient anchor |
| Train seconds | Thời gian nhánh train gồm audit; không gồm toàn bộ đánh giá cuối |

Tỷ lệ lỗi anchor trong train và flip-rate test khác tập và khác mẫu số. Bài bị cắt vẫn nằm trong metric test chính, không loại bỏ sau khi xem kết quả để làm đẹp accuracy.

## 7. Kết quả thực nghiệm

### 7.1. Tổng thể

| Metric | Ban đầu | GRPO | GRPO + ràng buộc |
|---|---:|---:|---:|
| Đúng / 640 test | 194 | 199 | 200 |
| Accuracy | 30,31% | 31,09% | 31,25% |
| Đúng → sai | - | 20 | 17 |
| Flip-rate / 194 bài ban đầu đúng | - | 10,31% | 8,76% |
| Sai → đúng | - | 25 | 23 |
| Validation accuracy | 28,13% | 27,50% | 28,13% |
| Worst topic | 21,25% | 23,75% | 24,38% |
| Test bị cắt | 2,50% | 3,13% | 3,44% |
| Test không box | 6,25% | 7,03% | 7,03% |
| Train, gồm audit | - | 3,32 giờ | 3,89 giờ |

```text
GRPO:       194 - 20 + 25 = 199
Ràng buộc:  194 - 17 + 23 = 200
```

Ít hơn 3 bài đúng → sai, nhưng cũng ít hơn 2 bài sai → đúng: **ròng hơn 1 bài**, tức **+0,156 điểm phần trăm accuracy**. Flip-rate giảm **1,546 điểm phần trăm**. Đây là chênh lệch tổng hợp; hai nhánh có thể đúng trên các câu khác nhau, không bảo đảm mọi bài GRPO giữ được đều được ràng buộc giữ lại.

### 7.2. Theo chủ đề

Mỗi chủ đề có 160 câu test; mẫu số flip là số câu ban đầu đúng trong chủ đề.

| Chủ đề | Accuracy ban đầu | GRPO | Ràng buộc | Số flip GRPO → ràng buộc | Flip-rate GRPO → ràng buộc |
|---|---:|---:|---:|---:|---:|
| Đại số | 51,88% | 51,25% | 51,25% | 9 → 7 / 83 | 10,84% → 8,43% |
| Hình học | 21,25% | 23,75% | 24,38% | 2 → 1 / 34 | 5,88% → 2,94% |
| Lý thuyết số | 21,88% | 24,38% | 25,00% | 4 → 4 / 35 | 11,43% → 11,43% |
| Tổ hợp/xác suất | 26,25% | 25,00% | 24,38% | 5 → 5 / 42 | 11,90% → 11,90% |

Giảm hồi quy tập trung ở đại số và hình học. Hai chủ đề còn lại không giảm flip; tổ hợp/xác suất còn thấp hơn GRPO một câu đúng. Chưa có lợi ích đồng đều trên các chủ đề.

### 7.3. Hoạt động train và chi phí

| Chỉ số | GRPO | Ràng buộc |
|---|---:|---:|
| Group RL / lời giải RL | 128 / 512 | 128 / 512 |
| Mixed groups | 49 (38,28%) | 43 (33,59%) |
| Token sinh lời giải RL | 149.416 | 148.054 |
| Token kiểm tra anchor trong train | 0 | 36.269 |
| Token audit trước train | 38.403 | 38.403 |
| Optimizer updates | 98 | 150 |
| Kiểm tra anchor trong train | 0 | 128 |
| Policy epoch có anchor loss | 0 | 96 |

Có 9 lượt anchor trả lời sai / 128 lượt kiểm tra = **7,03%**. Đây là quan sát ở nhiều thời điểm, không phải đánh giá toàn bộ 153 anchor tại checkpoint cuối; không thay thế flip-rate test 8,76%.

| λ sau bước | Đại số | Hình học | Lý thuyết số | Tổ hợp/xác suất |
|---|---:|---:|---:|---:|
| 32 | 0 | 0 | 0,0075 | 0,0075 |
| 64 | 0,0075 | 0 | 0 | 0 |
| 96 | 0,0150 | 0 | 0,0075 | 0,0075 |
| 128 | 0 | 0,0075 | 0,0150 | 0,0150 |

λ sau bước 32 dùng từ bước 33; λ sau bước 128 không được dùng thêm. Giá trị λ nhỏ không tự động chứng minh gradient anchor nhỏ tương ứng.

Train tăng **34,27 phút, khoảng 17,2%**. Tổng pipeline **15,85 giờ**, dùng lại cache suy luận ban đầu; không tính chi phí tạo cache ở run trước. Token RL không phản ánh toàn compute, vì còn sinh kiểm tra anchor và forward/backward trên gold solution.

### 7.4. Cắt token và độ bất định

Trên **cùng 615 câu không bị cắt ở cả ba model**, có 193 câu ban đầu đúng: GRPO có 19 flip, 24 bài sai → đúng; ràng buộc có 16 flip, 22 bài sai → đúng. Tín hiệu giảm flip vẫn còn. Tập được chọn theo đầu ra sau chạy nên chỉ là chẩn đoán phụ.

`paired_uncapped_test` trong CSV lọc từng cặp base/method, cho 617 câu với GRPO và 615 với ràng buộc. Phân tích trên dùng **giao chung cả ba model: 615 câu**, không so hai tập khác nhau.

Bootstrap 2.000 lần giữ cặp dự đoán, lấy mẫu câu trong từng chủ đề; chênh lệch **ràng buộc trừ GRPO**:

| Chênh lệch | Ước lượng | Khoảng tin cậy bootstrap 95% |
|---|---:|---:|
| Accuracy | +0,156 điểm phần trăm | [-0,781; +1,094] |
| Flip-rate | -1,546 điểm phần trăm | [-3,941; +0,532] |

Cả hai khoảng chứa 0: chưa loại trừ chênh lệch do mẫu câu. Một seed không đo được biến thiên training seed; bootstrap không thay thế nhiều lần train độc lập.

## 8. Đọc visualize_results.ipynb

Notebook có ba nhóm hình của lượt hiện tại:

1. **Bốn ô tổng thể:** accuracy (cao hơn tốt hơn), flip-rate (thấp hơn tốt hơn), train time và số bài đúng → sai/sai → đúng. Cần đọc cả flip và gain để thấy đánh đổi. Trục accuracy không bắt đầu từ 0, nên phải đọc nhãn số thay vì chỉ nhìn chiều cao cột.
2. **Hai ô theo chủ đề:** accuracy và tỷ lệ flip của từng chủ đề. Mẫu số flip khác nhau: 83/34/35/42, nên cùng một flip không tương ứng cùng tỷ lệ.
3. **Lịch λ:** trọng số sau mỗi bước, cập nhật mỗi 32 bước. Không phải loss hoặc flip-rate. Giá trị mới dùng ở bước sau; điểm cuối 128 không tạo thêm update.

Các bảng/chú thích nêu rõ một seed, test đã được xem trước và khoảng tin cậy chứa 0. Cột train time base bằng 0 vì model ban đầu không được train trong lượt này, không có nghĩa suy luận ban đầu miễn phí.

## 9. Kết luận và hạn chế

**Được hỗ trợ bởi run này:** số flip ít hơn 3 bài, accuracy gần tương đương, train tốn hơn. Đây là tín hiệu nhỏ về giữ đáp án đúng trong hậu huấn luyện đa chủ đề.

**Chưa được hỗ trợ:** vượt trội ổn định, không còn đúng → sai, tốt hơn mọi biến thể GRPO, mới về thuật toán hoặc giải quyết catastrophic forgetting. Repo hiện không có nhánh replay/CoKL/Dr. GRPO để kết luận về chúng.

Hạn chế:

- Một seed; subset đáp án nguyên không đại diện toàn bộ MATH.
- Test đã được xem trước khi quay lại feedback cửa sổ. Không dùng test trong gradient, nhưng lựa chọn hướng thử nghiệm đã chịu thông tin từ test: **đánh giá thăm dò hậu nghiệm**.
- Metric phụ thuộc boxed answer, token limit và decoding; không chấm tính đúng của từng bước suy luận.
- Cửa sổ chỉ có 8 lượt kiểm tra/chủ đề, có thể nhiễu; feedback trễ và gold loss là surrogate.
- Anchor dựa trên một lần giải đúng ban đầu, chưa xác nhận khả năng ổn định qua nhiều lần sinh.
- Hai nhánh khác compute/update; giảm flip đi cùng sửa ít bài sai hơn. Chưa tách tác động riêng của λ thích nghi khỏi tác động học thêm gold solution.

Nếu tiếp tục: dùng test chưa xem và nhiều seed; chốt config trước chạy; so thêm anchor loss có trọng số cố định để kiểm tra giá trị của feedback; khảo sát ngưỡng/cửa sổ trên validation; báo cáo đồng thời accuracy, flip, gain và compute. Đây là hướng phát triển, chưa phải kết quả đã có.

## 10. Cấu trúc và cách chạy

```text
configs/math_retention.json     Cấu hình của lượt được giữ
prepare_math_data.py            Kiểm tra frozen split đã có
train.py                       Train hai nhánh
src/plmco/math_trainer.py       Loss GRPO, cộng gradient anchor và train loop
src/plmco/math_model.py         Generation, chấm đáp án và gold loss
src/plmco/retention.py          Chọn anchor, cập nhật λ
src/plmco/modeling.py           QLoRA và precision thống nhất
src/plmco/math_data.py          Schema và kiểm tra split
src/plmco/paired_statistics.py  Bootstrap theo cặp
evaluate.py                    Đánh giá initial và adapter cuối
compare.py                     Metric và thống kê
screen_report.py               In bảng kết quả
run_replication.py             Runner toàn pipeline
visualize_results.ipynb        Bảng và biểu đồ
results/                       CSV, timing và lịch λ trên Git
output/pdf/                    Báo cáo PDF
outputs/                       Split, adapter, log, dự đoán chi tiết (local)
hf_cache/                      Cache model/dataset (local)
tests/                         Kiểm tra objective, feedback, audit, metric
```

### VS Code

Chọn `C:\Users\slywi\anaconda3\envs\tf_gpu\python.exe`. Code không thay đổi CUDA/TensorFlow. Config mặc định `configs/math_retention.json`; bỏ biến `PLMCO_CONFIG` cũ nếu có.

| Nhu cầu | File cần chạy |
|---|---|
| Xem kết quả | `visualize_results.ipynb` → **Run All** |
| Tạo lại bảng từ raw predictions local | `compare.py` → `screen_report.py` → notebook |
| Toàn pipeline với artifact local đã có | `run_replication.py` |
| Từng bước | `prepare_math_data.py` → `train.py` → `evaluate.py` → `compare.py` → `screen_report.py` → notebook |

Run đã hoàn tất: nhánh train/đánh giá hoàn tất được bỏ qua. Runner không tự train tiếp. Nhánh train bị ngắt sẽ khởi tạo lại, **không resume optimizer**. Muốn đổi cấu hình/mở rộng phải tạo run mới, không ghi đè run này bằng dữ liệu khác.

### Kết quả và khả năng tái lập

Local: `outputs/general_math_window_1024_seed42_deadline_oct6/` chứa frozen split, anchor ban đầu, initial evaluation; hai folder `seed_42/grpo/` và `seed_42/flip_constrained_grpo/` chứa adapter, summary, rollout, metrics, audit và evaluation. Plot nằm trong `notebook_plots/`.

GitHub có code, README, PDF, notebook với output đã lưu và bảng `results/`. **Không đưa model cache, adapter, frozen split hoặc raw predictions chi tiết lên Git.** Clone repo có thể xem báo cáo/chạy notebook từ `results/` mà không cần GPU; để train lại đúng tập cần artifact local đã giữ. `prepare_math_data.py` chỉ kiểm tra frozen split có sẵn, không tự tạo split thay thế khi thiếu.

Một số trường config lịch sử được giữ để bảo toàn hash frozen split, nhưng không dùng trong hai method hiện tại. Nguồn số liệu: `results/comparison.csv`, `results/paired_statistics.csv`, `results/retention_weights.csv`, `results/run_timing.json`. Lần cập nhật tài liệu này không thay đổi code train hoặc dữ liệu kết quả.
