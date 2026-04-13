# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Ngọc Thắng  
**Mã SV:** 2A202600191  
**Vai trò trong nhóm:** Generation / RAG Pipeline Owner (từ Query Transform)  
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
Với câu hỏi này, mục tiêu của pipeline là trả về được đúng thời hạn (số ngày cụ thể) tính từ lúc nào, đồng thời đính kèm bằng chứng ví dụ từ tài liệu Policy. Khi dùng luồng tìm kiếm dense retrieval cũ, query này đôi khi trượt mất do từ vựng có thể không map chặt chẽ (vd trong doc nó nằm dưới dạng "refund timeframe", "return SLA"). Nhưng qua hệ thống tôi triển khai với bước Query Expansion ở `transform_query`, câu truy vấn đã được đẻ thêm các biến thể từ lóng/đồng nghĩa và ghép vào chung để đi kiếm. Kết quả thu gộp sẽ dồi dào hơn ở khâu retrieve đầu vào. Đồng thời, lúc build prompt, văn bản cung cấp thời hạn hoàn tiền sẽ được gài gọn vô `build_context_block` với số thứ tự `[1]`. Nhờ thiết kế Prompt gắt gao với quy định "chỉ giải đáp nếu thông tin tồn tại trong văn bản", LLM chỉ bám chặt với dữ liệu có trong khối tài liệu đó để nói ra "30 ngày" (hay mức ngày nào đó từ doc), đi cùng dòng chữ dẫn xuất `[1]`. Cơ chế này loại trừ luôn khả năng model tự áp đặt kiến thức 14 ngày/7 ngày nào đó sẵn có trong data pretrain của nó.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi dự kiến tinh giảm các lượt call API bằng việc xây dựng bộ nhớ Cache cho các query transformation trùng nhau. Tôi cũng muốn cài cắm cơ chế Self-RAG - một hệ thống con giúp mô hình tự check/chấm điểm bản thân một lần trước khi đưa kết quả để giảm thiểu đáng kể số lần Generate "rỗng" hoặc bị lạc đề, nâng cao chất lượng báo cáo cho scorecard khi đi vào đánh giá Evaluation tự động.

---
