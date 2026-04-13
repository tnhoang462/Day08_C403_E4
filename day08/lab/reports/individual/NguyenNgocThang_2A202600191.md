# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Ngọc Thắng  
**Mã SV:** 2A202600191  
**Vai trò trong nhóm:** Tech Lead & RAG Pipeline Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, nhiệm vụ chính của tôi tập trung vào file `rag_answer.py`, cụ thể là phần triển khai chiến lược Query Transformation, xây dựng ngữ cảnh (context) và Generation. Đầu tiên, tôi tiến hành xử lý đầu vào của người dùng thông qua hàm `transform_query` bằng các phương pháp như query expansion (sinh thêm các biến thể tìm kiếm), decomposition (chia nhỏ câu hỏi phức tạp) và HyDE (tạo văn bản giải quyết câu hỏi giả định) nhằm tăng scope recall trước khi retrieval. Kế tiếp, tôi xây dựng cụm hàm xử lý đầu ra: gộp các chunk kết quả qua hàm `build_context_block` dưới dạng metadata tường minh (`[1] source | section`) để tạo thuận lợi cho model trích dẫn nguồn. Tôi cũng định hình `build_grounded_prompt` để yêu cầu LLM có tính chất "evidence-only", cam kết báo "từ chối" nếu không có thông tin. Cuối cùng, tôi dựng hàm tích hợp luồng `rag_answer` và script `compare_retrieval_strategies` để phục vụ cho các phép so sánh A/B test hiệu quả của RAG.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Qua quá trình thực hiện các khâu thuộc Generation và Query Manipulation, tôi nhận thức sâu sắc hơn về thiết kế Prompts sao cho "tư duy RAG" nhất quán với "tư duy LLMs". Trước đây, tôi tưởng chỉ cần một câu hỏi truyền vào kèm context dài thòng là LLMs sẽ tự moi thông tin. Nhưng thực tế nếu đầu vào không tổ chức tốt và không ép citation thì model sinh chữ tào lao (hallucination) ngay lập tức. Cú pháp truyền như `[1] DocsName | Section` trong `build_context_block` giúp LLMs phân trang và móc nối chính xác. Mặt khác, áp dụng query transformation đã khai sáng cho tôi một góc rễ mới: đôi khi lỗi không nằm ở việc vector search ngu đần, mà bản thân câu hỏi của con người vốn thiếu mất bối cảnh trọng tâm. Sự tiền xử lý query tạo tiền đề lớn cho sự thành bại của Retrieval. 

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Trong lúc thực thi, thứ ngốn thời gian mà tôi không ngờ nhất là khả năng model không tuân theo đúng định dạng JSON hoặc format trong bước `transform_query`. Đôi khi chúng trả về đoạn text chứa markdown code block (` ```json `), ép tôi phải code thêm ngoại lệ tự động parse JSON an toàn thay vì dùng raw string. Bên cạnh đó, việc tìm điểm thăng bằng để thiết lập system prompt cho tính chất (abstain) cũng gây lúng túng. Nếu khắt khe quá mức, mô hình không dám trả lời kể cả khi thông tin nằm sẵn đó; lơi lỏng một xíu lại chém gió. Thêm nữa, khi áp dụng chiến lược HyDE (viết 1 đoạn giải đáp giả lập và đem đi embed lại), sự thay đổi này làm độ trễ response tăng vọt (vì tốn call api tới LLM hai lượt), nhưng chất lượng Context Chunk trả lùi về cũng đồng thời tăng rất bất ngờ.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?"

**Phân tích:**
Failure Mode chính: Generation — Incomplete Synthesis (thiếu tổng hợp cross-document)
Câu trả lời hệ thống chỉ đưa ra "tối đa 2 thiết bị cùng lúc [1]" — đúng nhưng thiếu nghiêm trọng. Expected answer yêu cầu tổng hợp từ 2 nguồn: HR Leave Policy (VPN bắt buộc khi remote) và IT Helpdesk FAQ (Cisco AnyConnect, giới hạn 2 thiết bị). Completeness chỉ đạt 2/5.
Root Cause Trace:
Retrieval thực tế không lỗi — context_recall đạt 5/5, nghĩa là pipeline retrieve_dense() đã lấy đủ 2 chunk từ 2 tài liệu. Vấn đề nằm ở tầng Generation: hàm build_grounded_prompt() trong rag_answer.py chỉ yêu cầu LLM "trả lời ngắn gọn" (Keep your answer short, clear) mà không có chỉ dẫn tổng hợp đa nguồn. LLM đã "lười" — chỉ trích 1 chunk dễ nhất rồi dừng.
Đề xuất fix cụ thể:

Sửa prompt trong build_grounded_prompt(): Thêm instruction "If multiple sources are relevant, synthesize information from ALL of them and cite each source separately."
Thêm multi-hop detection: Trước khi gọi LLM, kiểm tra nếu câu hỏi chứa nhiều sub-question (ở đây: "phải dùng VPN" + "tối đa bao nhiêu thiết bị"), dùng transform_query("decomposition") để tách query, đảm bảo mỗi phần được trả lời.
Post-check: So sánh số source trong answer vs số chunk retrieved — nếu chênh lệch lớn, cảnh báo thiếu thông tin.
---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi dự kiến tinh giảm các lượt call API bằng việc xây dựng bộ nhớ Cache cho các query transformation trùng nhau. Tôi cũng muốn cài cắm cơ chế Self-RAG - một hệ thống con giúp mô hình tự check/chấm điểm bản thân một lần trước khi đưa kết quả để giảm thiểu đáng kể số lần Generate "rỗng" hoặc bị lạc đề, nâng cao chất lượng báo cáo cho scorecard khi đi vào đánh giá Evaluation tự động.

---
