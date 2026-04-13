# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Nhật Hoàng  
**Vai trò trong nhóm:** Retrieval Owner / Debug Support  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, phần tôi phụ trách chính nằm ở **Sprint 3**, tập trung trong file `rag_answer.py`, đặc biệt là hai hàm `retrieve_hybrid()` và `rerank()`. Ở `retrieve_hybrid()`, tôi phụ trách cách kết hợp dense retrieval và sparse retrieval bằng **Reciprocal Rank Fusion (RRF)** để vừa giữ được ngữ nghĩa, vừa không bỏ sót keyword kỹ thuật. Quyết định này phù hợp với bộ tài liệu nội bộ có cả câu văn tự nhiên, tên cũ của tài liệu, mã lỗi và điều khoản. Bên cạnh phần tuning retrieval, tôi cũng hỗ trợ **debug các hàm khác trong `rag_answer.py`** như luồng retrieve, deduplicate candidate chunks, sort theo score và nối retrieval với generation.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, điều tôi hiểu rõ hơn là sự khác biệt giữa **dense retrieval**, **hybrid retrieval** và **reranking**. Dense retrieval mạnh ở việc hiểu ý nghĩa tổng quát của câu hỏi, nhưng dễ hụt alias, tên cũ hoặc keyword ngắn. Hybrid retrieval bù điểm yếu đó bằng cách lấy thêm tín hiệu từ sparse search rồi hợp nhất thứ hạng qua RRF. Còn rerank không thay retrieval, mà là bước chấm lại để chọn ra vài chunk tốt nhất trước khi đưa vào prompt. Từ quá trình này, tôi hiểu rõ hơn rằng chất lượng RAG không phụ thuộc riêng vào model sinh câu trả lời; phần quyết định rất lớn nằm ở việc retrieve đúng context và chọn đúng chunk để model dựa vào.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất của tôi là nhận ra rằng **debug RAG không thể chỉ nhìn câu trả lời cuối cùng**, mà phải tách theo từng tầng: retrieve được gì, sort ra sao, chọn chunk nào, rồi LLM mới trả lời thế nào. Ban đầu tôi nghĩ chỉ cần thêm hybrid và rerank thì score sẽ tăng đồng loạt, nhưng thực tế không đơn giản như vậy. Có những câu retrieval đã lấy đúng source, nhưng câu trả lời vẫn chưa thật sự grounded vì generation còn suy diễn thêm. Ngoài ra, nhiều lỗi trong `rag_answer.py` không nổ exception rõ ràng mà chỉ thể hiện qua hành vi “gần đúng”, như dedup chưa hợp lý hoặc thứ tự score sau hợp nhất chưa chuẩn.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** “Khi làm việc remote, tôi phải dùng VPN và được kết nối trên tối đa bao nhiêu thiết bị?” (`gq02`)

**Phân tích:**

Đây là câu hỏi tôi thấy thú vị nhất vì nó cho thấy rõ ranh giới giữa retrieval và generation. Ở **Baseline**, scorecard cho thấy `Context Recall = 5/5`, `Relevance = 5/5`, nhưng `Faithfulness = 3/5`. Điều đó nghĩa là hệ thống đã retrieve đúng tài liệu liên quan, nhưng câu trả lời vẫn thêm vài chi tiết chưa bám chặt hoàn toàn vào context. Khi sang **Variant (hybrid + rerank)**, `Context Recall` vẫn là `5/5`, `Relevance` vẫn `5/5`, nhưng `Faithfulness` giảm xuống `2/5`. Như vậy, riêng câu này variant không cải thiện kết quả cuối. Theo tôi, lỗi chính không nằm ở indexing vì nguồn cần thiết đã được lấy đúng ở cả hai phiên bản. Retrieval cũng không phải điểm nghẽn lớn nhất vì recall đã tối đa. Điểm khó của câu này là nó thuộc dạng **cross-document**: mô hình phải ghép yêu cầu dùng VPN với giới hạn số thiết bị từ nhiều nguồn. Hybrid và rerank chỉ giúp tăng cơ hội chọn đúng context; nếu prompt generation chưa đủ chặt thì LLM vẫn có thể suy diễn thêm. Trường hợp này giúp tôi hiểu rằng tuning retrieval là cần, nhưng chưa đủ để tăng faithfulness ổn định.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ hoàn thiện `retrieve_sparse()` để hybrid chạy đúng nghĩa end-to-end, thay vì mới mạnh ở phần khung hợp nhất trong `retrieve_hybrid()`. Sau đó tôi muốn triển khai `rerank()` bằng một cross-encoder thật sự thay cho bản placeholder hiện tại. Lý do là scorecard cho thấy recall đã cao nhưng faithfulness chưa ổn định, nên tôi muốn tối ưu mạnh hơn ở bước chọn context đầu vào cho prompt.
