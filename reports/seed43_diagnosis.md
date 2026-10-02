# Kiểm tra seed 43 và bản sửa feedback v2

Ngày kiểm tra: 02/10/2026. Không chạy huấn luyện mới; chỉ đọc artifact v1, kiểm tra code, unit test và suy luận ba lần trên một anchor.

## Những gì có bằng chứng

1. Seed 43: GRPO có 8/47 bài test ban đầu đúng chuyển thành sai; ràng buộc có 10/47. Ràng buộc vẫn đúng tổng cộng 45 bài, hơn GRPO 44 bài, nhờ sửa đúng nhiều bài ban đầu sai hơn. Vì vậy seed 43 thất bại ở mục tiêu giảm flip, không phải thua mọi metric.
2. Hệ số ràng buộc khởi tạo bằng 0 và chỉ cập nhật sau bước 32. Anchor đầu tiên báo sai ở bước 4 của seed 43; 11/32 kiểm tra trong nửa đầu báo sai. Các hệ số chỉ thực sự được dùng ở bước 33–64. Lần tăng hệ số sau bước 64 không tác động đến bất kỳ update tiếp theo nào.
3. Tập anchor-correct v1 chỉ có 17 bài: đại số 7, hình học 2, lý thuyết số 4, tổ hợp/xác suất 4. Mỗi lần kiểm tra lại vẫn được đếm thêm trong tỷ lệ window. Độ phủ nhỏ, đặc biệt ở hình học, hạn chế khả năng bảo vệ các câu test khác.
4. Có **precision mismatch** giữa loader train và evaluate. Train gọi `prepare_model_for_kbit_training`, cast các tham số FP16/BF16 không phải `Params4bit` sang FP32; loader inference v1 bỏ qua bước này. Quy tắc đó được kiểm tra trong bản PEFT đã cài tại `tf_gpu`.

## Kiểm chứng precision bằng suy luận, không có optimizer update

Anchor `geometry:train:350` hỏi giá trị tan(225°), đáp án đúng là 1. Dùng cùng checkpoint Qwen2.5-1.5B-Instruct, quantization, tokenizer và seed 43, không train:

| Loader | Embedding dtype | Đáp án | Đúng? | Token |
|---|---|---:|---|---:|
| Inference v1 | BF16 | 1 | Có | 128 |
| Model vừa chuẩn bị QLoRA, LoRA chưa cập nhật | FP32 | −1 | Không | 99 |
| Inference v2 đồng nhất với chuẩn bị QLoRA | FP32 | −1 | Không | 99 |

[JSON kiểm tra](seed43_precision_diagnosis.json) lưu cả lời giải. Đây là bằng chứng trực tiếp rằng ít nhất một anchor bị gọi là “flip do train” chỉ do đổi cách nạp model. Log seed 42 cũng báo ba flip tại bước 2–4, trước mixed group đầu tiên ở bước 7 và trước mọi optimizer update. Không thể gán các flip đó cho GRPO.

Điều này **không chứng minh precision mismatch là nguyên nhân duy nhất của 10 flip test ở seed 43**. Các đánh giá initial và adapter v1 cùng dùng loader inference v1; tuy nhiên train diễn ra trên một numerical base khác, nên cả phép đo anchor và diễn giải “cùng base trước/sau train” cần được kiểm chứng lại. Tác động của thứ tự bài, sampling và anchor loss lên seed 43 chưa được tách bằng ablation. Không được nói đã sửa xong hiệu năng khi chưa train v2.

## Bản sửa

- Giữ nguyên câu train/anchor/validation/test, model, reward, ngân sách RL và trần token.
- Đồng nhất numerical base khi scan anchor, initial, train và evaluate adapter. Không dùng lại initial predictions hay anchor-correct v1.
- Kiểm tra tất cả anchor được chọn trên fresh zero-LoRA trước train ở cả hai nhánh. Nếu khác baseline, dừng và lưu audit.
- Cập nhật λ sau quan sát anchor và trước optimizer update của chính bước đó. Bản v2 ban đầu dùng tối đa 4 bài khác nhau theo chủ đề; lượt mở rộng đã tăng lên 16. Không đếm một UID nhiều lần trong cùng ước lượng; vẫn có độ trễ do các anchor không được kiểm tra đồng thời.
- Lưu trọng số thực dùng, mẫu số feedback, anchor gold loss, số update và chi phí audit.
- Kế hoạch đã chốt lại: train độc lập cặp GRPO/ràng buộc v2 ở cả seed 42, 43 và 44, lưu lượt mới. Không chỉ rerun nhánh ràng buộc rồi so với GRPO v1 có chế độ đánh giá khác.

Đây là sửa tính nhất quán của baseline và phản hồi chậm, không phải một bảo đảm giảm đúng → sai. Gold loss vẫn chỉ là surrogate. Kế hoạch hiện tại đã mở rộng thành 256 train, 512 anchor ứng viên, 160 validation và 640 test, loại toàn bộ đề của split v1. Ba seed dùng chung test mới, không tính các lượt đánh giá đó như những câu độc lập. Lượt này đánh giá v2 so với GRPO, chưa phải ablation v2 so với v1 dưới cùng precision.
