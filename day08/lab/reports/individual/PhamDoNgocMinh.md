# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phạm Đỗ Ngọc Minh
**Vai trò trong nhóm:** Tech Lead / Eval Owner / Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhận vai trò Tech Lead kết hợp Eval và Documentation Owner. Phần lớn thời gian của tôi tập trung vào Sprint 1, Sprint 4 và hoàn thiện hệ thống Docs. Cụ thể, tôi đã hoàn thiện `eval.py` để triển khai LLM-as-a-Judge, đánh giá RAG pipeline trên 4 metrics: Faithfulness, Answer Relevance, Context Recall và Completeness. Tôi cũng điều chỉnh `index.py` để đảm bảo khâu băm tài liệu (chunking) xử lý hiệu quả. 
Bên cạnh đó, tôi dành công sức rà soát lỗi từ pipeline (sửa lỗi thiếu file JSON test questions) và chạy A/B comparison giữa Baseline (Dense) và Variant (Hybrid + Rerank). Tôi đã ghi lại Architecture Design và Tuning-log để lưu toàn bộ thực nghiệm. Công việc của tôi kết nối trực tiếp với Sprint 2 & 3 của team: chỉ khi có bộ Evaluation mạnh (thong qua metric số hóa) và file kiến trúc đủ chi tiết, team mới chứng minh được việc đổi sang Hybrid Search kèm Cross-Encoder Rerank là một quyết định cải thiện chất lượng đúng đắn.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Concept tôi nhận thấy mình hiểu rõ nhất sau lab này là bộ khung **Evaluation Loop bằng LLM-as-a-Judge** và **Hybrid Retrieval**.
Ban đầu, tôi không hình dung rõ sự khác nhau giữa relevance và faithfulness, nhưng qua quá trình giám sát log từ `eval.py`, tôi nhận ra bộ metric cực kỳ nhạy bén trong việc đánh lỗi hallucination. Nếu LLM Generation bị thiếu instruction kìm hãm (Grounded Prompt), nó lập tức tự tạo ra kiến thức ảo (điểm Faithfulness rớt ngay).
Thứ hai, tôi thực sự nắm được giá trị của Hybrid Retrieval. Dense Search rất tốt ở khả năng hiểu ngữ nghĩa chung chung, nhưng lại bỏ sót khi người dùng gõ từ khóa kỹ thuật như "ERR-403-AUTH" hay "SLA P1". Việc tích hợp Sparse Search (BM25) vào và dung hoà thông qua thuật toán Reciprocal Rank Fusion (RRF) là mảnh ghép hoàn hảo để khắc phục điểm mù của Dense Vector.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất tôi gặp phải là việc hệ thống evaluation ban đầu không map đúng file questions (`grading_questions.json` bị sai tên gốc), khiến pipeline `eval.py` ném ra lỗi file not found và sập script ở khâu load testing. Hơn thế nữa, module biến thể `hybrid` chưa được hoàn thiện ở file `rag_answer.py` nội bộ và phải sát nhập từ thay đổi của repository trước đó. Phải mất một lúc rà soát lại các luồng code và debug thì tôi mới khắc phục được.
Điều khiến tôi ngạc nhiên nhất là **độ "ngáo" (hallucination) của LLM khi làm Baseline**. Mặc dù đã được cung cấp Context đầy đủ, mô hình lập tức lấy kiến thức cũ từ pre-training để trả lời bịa đặt nếu câu hỏi đánh mẹo. Hệ thống RAG thực sự cần quá trình Strict Prompting (quy tắc persona gắt gao & force citation) kết hợp cùng Rerank chặt chẽ mới có thể an tâm cho user sử dụng.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** [gq07] Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?

**Phân tích:**
Đây là một câu hỏi bẫy test năng lực Abstain ("Insufficient Context"). Trong kho tài liệu hoàn toàn không có văn bản nào quy định số tiền phạt hay mức kỷ luật cho việc vi phạm SLA P1. 
- Ở **Baseline (Dense)**: Mô hình ngây ngô không biết cách từ chối trả lời. Thấy chữ SLA P1, hệ thống retrieve về tài liệu chính sách chung chung, và LLM tự sinh ra một số tiền phạt ảo tưởng. Điểm Faithfulness tụt chạm đáy (1/5). Lỗi ở đây nằm ở **Generation** (System prompt Baseline quá hời hợt, cho phép model chém gió).
- Ở **Variant (Hybrid + Rerank + Cải tiến Prompt)**: Pipeline đã nhận diện được bẫy, LLM trả về đúng: *"Tôi không tìm thấy thông tin này trong tài liệu nội bộ hiện có."* Lỗi hallucination bị dập tắt (Faithfulness tăng vọt). Biến thể này tốt hơn hẳn ko phải nhờ Retrieval (vì vốn không có chunk đáp án cho câu này), mà chiến thắng là thiết kế của hệ thống **Grounded Prompting** khắt khe bắt model "do not fabricate if context is insufficient".

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ thử nghiệm thêm module **Query Transformation (Decomposition và HyDE)** thay vì chỉ loanh quanh Sparse/Dense. Kết quả eval chỉ ra rằng một số câu hỏi ghép nhiều vế song song (vd: `gq08` vừa hỏi điều kiện phép năm, vừa hỏi thủ tục ngày nghỉ ốm) rất dễ làm hệ thống lấy nhầm chunk. Việc tách đôi truy vấn thành biểu đồ đa nhánh và gộp đáp án sau này chắn chắn sẽ vá mọi lỗ hổng ở Completeness score.
