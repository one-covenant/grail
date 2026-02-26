# Hướng dẫn đọc codebase GRAIL cho người mới

Tài liệu này giúp bạn hiểu **mã gốc** theo hướng “đi từ tổng quan đến chi tiết”, không cần biết hết Bittensor hay RL ngay từ đầu.

## 1) Cấu trúc chung của hệ thống

Ở mức cao, dự án này có 2 vai trò chạy song song:

- **Miner**: sinh rollout (output mô hình + metadata + proof), upload lên object storage.
- **Validator**: tải rollout của miner, kiểm tra tính hợp lệ qua pipeline, chấm điểm và cập nhật weight.

Entry point chạy qua CLI `grail`:

- `python -m grail` gọi vào CLI chính trong `grail/cli/__init__.py`.
- CLI đăng ký các lệnh như `mine`, `validate`, `train`.

Nói ngắn gọn: **CLI → Neuron (vòng đời node) → Service/Pipeline (nghiệp vụ) → Protocol/Infrastructure (proof, chain, storage, randomness).**

## 2) “Map” thư mục nên biết trước

- `grail/cli/`: giao diện lệnh và logging khởi động.
- `grail/neurons/`: vòng đời miner/validator, xử lý signal shutdown, watchdog, heartbeat.
- `grail/validation/`: pipeline validator (schema/token/proof/env/reward/logprob...), xử lý từng window.
- `grail/protocol/`: primitive crypto, token hashing, signature/commit binding, verifier logic.
- `grail/infrastructure/`: chain manager, credentials, comms object storage, checkpoint consumer.
- `grail/scoring/`: tính toán weight dựa trên điểm/độ ổn định.
- `grail/shared/`: constants và utility dùng toàn hệ thống.
- `docs/`: tài liệu vận hành miner/validator.

## 3) Luồng chạy thực tế (mental model)

### 3.1 Miner

1. Khởi tạo wallet + kết nối chain.
2. Tải credentials R2/S3, chuẩn bị checkpoint model phù hợp window.
3. Kết hợp randomness (block hash + drand nếu bật).
4. Sinh rollout cho batch bài toán môi trường.
5. Đóng gói output, ký/proof, upload file window.
6. Lặp theo window mới.

Điểm hay là miner dùng `BaseNeuron` nên có cơ chế chung: signal handling, heartbeat, watchdog tránh treo im lặng.

### 3.2 Validator

1. Khởi tạo `ValidationService` với dependency rõ ràng (wallet/netuid/pipeline/weight computer/checkpoint manager...).
2. Theo dõi block hiện tại để xác định window đã “đóng” và cần chấm.
3. Lấy rollout miner, tạo `ValidationContext`, chạy pipeline tuần tự.
4. Nếu pass các check cứng (hard checks), cộng vào thống kê chấm điểm.
5. Định kỳ tính weight rolling và submit on-chain.

Thiết kế pipeline là fail-fast cho check cứng: fail thì dừng sớm, tiết kiệm compute.

## 4) Những điều quan trọng cần nhớ

### 4.1 Window-based processing là xương sống

Hầu hết logic đều xoay quanh “window” (tập block):

- miner sinh dữ liệu theo window,
- validator chấm window đã hoàn tất,
- scoring dùng rolling windows.

Vì vậy, nếu muốn debug đúng chỗ, hãy luôn hỏi: “Mình đang ở window nào?”

### 4.2 Validation có “hard check” và “soft check”

- **Hard check** fail ⇒ rollout không hợp lệ.
- **Soft check** fail ⇒ vẫn có thể tiếp tục các check khác (dùng làm heuristic/chất lượng).

Điều này quyết định trực tiếp miner có được tính điểm hay không.

### 4.3 Constants quyết định hành vi hệ thống

Các giá trị như `WINDOW_LENGTH`, `CHALLENGE_K`, `PROOF_TOPK`, `MINER_SAMPLE_RATE`... nằm ở `grail/shared/constants.py` và ảnh hưởng lớn đến:

- nhịp xử lý,
- độ nghiêm của proof,
- chi phí validate,
- cách phân phối weight.

Khi đọc bug/perf, hãy kiểm tra constants trước khi nghi ngờ thuật toán phức tạp.

### 4.4 Protocol được tách module, không còn “1 file khổng lồ”

`grail/protocol/` tách ra thành crypto/signatures/tokens/verifier để dễ test và thay thế thành phần. Đây là dấu hiệu kiến trúc đã hướng tới maintainability tốt hơn.

## 5) Lộ trình học tiếp theo (gợi ý cho người mới)

## Giai đoạn A: Nắm khung chạy

1. Đọc `README.md` để hiểu mục tiêu subnet.
2. Đọc `docs/miner.md` và `docs/validator.md` để nắm vận hành.
3. Chạy `grail --help`, `grail mine --help`, `grail validate --help`.

Mục tiêu: biết hệ thống làm gì trước khi biết “làm thế nào”.

## Giai đoạn B: Nắm luồng validate (quan trọng nhất)

1. Đọc `grail/validation/service.py` trước.
2. Đọc `grail/validation/pipeline.py` để hiểu thứ tự validator.
3. Mở từng file trong `grail/validation/validators/` (schema, token, proof, termination, environment...).

Mục tiêu: hiểu tiêu chí pass/fail của rollout.

## Giai đoạn C: Nắm protocol và crypto ở mức ứng dụng

1. Đọc `grail/protocol/__init__.py` để biết public API.
2. Đọc `grail/protocol/crypto.py`, `signatures.py`, `tokens.py`.
3. Đọc `grail/protocol/grail_verifier.py` để hiểu kiểm tra hidden-state sketch.

Mục tiêu: hiểu “proof đang chứng minh điều gì” thay vì chỉ chạy code.

## Giai đoạn D: Nắm scoring + economics

1. Đọc `grail/scoring/weights.py`.
2. Theo dõi cách validator gom metrics theo rolling window.
3. Map lại từ metric → score → weight on-chain.

Mục tiêu: biết vì sao miner được thưởng/phạt.

## 6) Cách đọc code nhanh mà đỡ ngợp

- Bắt đầu từ file có “service/orchestrator”, chưa đi sâu util ngay.
- Vẽ sơ đồ 1 vòng lặp đầy đủ: input → transform → output → state update.
- Mỗi lần đọc chỉ trả lời 3 câu:
  1. Hàm này nhận gì?
  2. Nó thay đổi state nào?
  3. Nó fail thì hệ thống phản ứng gì?
- Song song mở test integration liên quan để thấy “expected behavior” thực tế.

## 7) Checklist tự tin trước khi sửa code

- Bạn hiểu rõ dữ liệu theo window được tạo/lưu/chấm thế nào.
- Bạn biết validator nào là hard check, validator nào soft check.
- Bạn biết constants nào ảnh hưởng trực tiếp đến thay đổi của mình.
- Bạn chạy được ít nhất một test unit + một test integration liên quan vùng sửa.

---

Nếu bạn mới hoàn toàn, hãy ưu tiên hiểu **validation pipeline + constants + vòng đời window** trước. Đó là 20% kiến thức tạo ra ~80% khả năng debug/thêm tính năng trong codebase này.
