# Báo Cáo Cá Nhân - Trương Đặng Gia Huy (2A202600436)

**Vai trò:** Eval Owner (được phân công, chưa thực hiện)  
**Ngày nộp:** 14/04/2026

---

## 1. Đóng góp cụ thể

Em không đóng góp code, documentation, hay bất kỳ phần nào trong lab này. Em vắng mặt trong buổi lab do có chuyến bay sang Singapore cùng ngày để đi học. Toàn bộ pipeline, từ indexing, retrieval, generation đến evaluation, đều do các thành viên còn lại trong nhóm hoàn thành. Em không có commit nào trong repo và không tham gia thảo luận kỹ thuật trong quá trình làm bài.

Sau khi lab kết thúc, em đã đọc lại toàn bộ code (`index.py`, `rag_answer.py`, `eval.py`), kết quả eval (`eval-run-20260413-185950.json`, scorecard baseline và variant), cùng tài liệu nhóm (`architecture.md`, `tuning-log.md`) để hiểu pipeline nhóm đã xây dựng và tự học từ kết quả.

---

## 2. Phân tích câu grading: gq07 - Abstain / Anti-hallucination

**Câu hỏi:** "Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?"

**Expected:** Tài liệu không có thông tin về mức phạt, pipeline phải từ chối trả lời (abstain).

### Baseline xử lý thế nào

Ở scorecard baseline, gq07 nhận Faithfulness = 1/5 và Relevance = 1/5, điểm thấp nhất trong 10 câu. Pipeline baseline không nhận ra rằng tài liệu `sla_p1_2026.txt` chỉ mô tả quy trình xử lý và SLA target, không có điều khoản penalty. LLM tự suy luận từ general knowledge và đưa ra thông tin không có trong context. Đây là lỗi hallucination, lỗi nghiêm trọng nhất trong RAG pipeline vì nó tạo ra thông tin sai mà người dùng có thể tin là thật.

**Root cause:** Lỗi nằm ở tầng **Generation**. Context Recall không áp dụng (vì không có expected source), nhưng vấn đề cốt lõi là system prompt baseline thiếu instruction bắt buộc model abstain khi context không đủ. Prompt không có rule "chỉ trả lời dựa trên tài liệu" khiến LLM mặc định cố gắng trả lời bằng mọi giá.

### Variant xử lý thế nào

Ở scorecard variant (Hybrid + Rerank + Grounded Prompt), gq07 đạt Faithfulness = 5/5. Pipeline trả lời: *"Tôi không tìm thấy thông tin này trong tài liệu nội bộ hiện có."* Đây là kết quả của việc thêm strict grounding constraints vào prompt, rule buộc model từ chối trả lời nếu context không chứa thông tin, thay vì tự sáng tác.

### Đánh giá

Sự cải thiện từ 1/5 lên 5/5 Faithfulness cho thấy biến có tác động lớn nhất là **prompt engineering**, không phải retrieval mode hay reranker. Một dòng instruction đã biến pipeline từ "bịa đặt nguy hiểm" thành "từ chối an toàn". Relevance vẫn chỉ đạt 1/5 vì câu trả lời abstain đương nhiên không "trả lời" câu hỏi, nhưng đây là trade-off đúng đắn: thà không trả lời còn hơn trả lời sai.

---

## 3. Rút kinh nghiệm

Điều khiến em ngạc nhiên nhất khi đọc lại kết quả là variant (Hybrid + Rerank) bị crash toàn bộ 10 câu với lỗi `'NoneType' object is not iterable` trong file `eval-run-20260413-185950.json`. Pipeline tự động end-to-end của variant không hoạt động. Các scorecard variant có điểm số là vì nhóm chạy riêng từng câu hoặc sửa lỗi giữa chừng. Điều này cho thấy integration testing quan trọng không kém unit testing: mỗi module có thể chạy đúng riêng lẻ nhưng khi ghép lại thì fail ở chỗ truyền dữ liệu giữa các bước.

Bài học lớn nhất với cá nhân em là về trách nhiệm với nhóm. Em biết trước sẽ vắng mặt nhưng không chủ động nhận phần việc có thể làm async trước buổi lab (ví dụ viết test questions, chuẩn bị documentation template). Nếu làm lại, em sẽ hoàn thành phần việc có thể làm trước thay vì để nhóm thiếu người trong buổi thực hành.

---

## 4. Đề xuất cải tiến

**1. Query Decomposition cho câu cross-document:** Scorecard baseline cho thấy gq05 (Contractor Admin Access) chỉ đạt Completeness 1/5. Pipeline trả lời sai rằng contractor không được cấp Admin Access, trong khi tài liệu Access Control SOP ghi rõ quy trình áp dụng cho cả contractor và third-party vendor. Câu hỏi có nhiều vế (có được cấp không? bao nhiêu ngày? yêu cầu đặc biệt?) nhưng retrieval chỉ bắt được chunk liên quan đến một vế. Cải tiến: thêm bước decomposition tách câu hỏi thành 2-3 sub-queries, retrieve riêng từng sub-query, rồi tổng hợp context trước khi đưa vào LLM.

**2. Fix integration bug của variant:** Eval log cho thấy variant crash 10/10 câu với lỗi `NoneType`. Nhóm cần thêm null-check ở chỗ nối output của hybrid retrieval vào reranker. Nếu không fix, mọi cải tiến retrieval đều vô nghĩa vì pipeline không chạy được end-to-end để đo lường chính xác.

---
