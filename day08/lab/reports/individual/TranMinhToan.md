# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Minh Toàn  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/04  

## 1. Tôi đã làm gì trong lab này? 

Trong lab này, tôi phụ trách phần retrieval, cụ thể là xây dựng và tinh chỉnh chiến lược kết hợp Dense retrieval và Sparse/BM25. Ở sprint đầu, tôi tập trung chuẩn hóa dữ liệu đầu vào cho indexing, đảm bảo mỗi chunk giữ đủ ngữ cảnh và metadata để truy hồi ổn định. Sau đó, tôi implement hai nhánh truy hồi riêng: Dense cho semantic matching và BM25 cho keyword exact match. Ở sprint tiếp theo, tôi phối hợp với bạn phụ trách generation/eval để hợp nhất kết quả theo dạng hybrid retrieval và quan sát tác động lên scorecard. Tôi cũng thử các giá trị top-k và tỷ trọng giữa dense-sparse để giảm bỏ sót tài liệu quan trọng. Công việc của tôi kết nối trực tiếp với phần prompt/generation vì chất lượng context retrieved quyết định độ chính xác câu trả lời cuối cùng.

## 2. Điều tôi hiểu rõ hơn sau lab này 

Sau lab, tôi hiểu rõ hơn sự khác nhau thực tế giữa Dense và BM25 thay vì chỉ hiểu ở mức khái niệm. Dense retrieval mạnh khi câu hỏi diễn đạt linh hoạt hoặc dùng từ đồng nghĩa, vì embedding có thể bắt được nghĩa gần nhau. Ngược lại, BM25 lại rất hiệu quả với câu hỏi chứa cụm từ khóa đặc thù như mã SLA, phiên bản policy, hoặc thuật ngữ nội bộ. Tôi cũng hiểu rõ “hybrid retrieval” không chỉ là cộng hai danh sách kết quả, mà cần có chiến lược cân bằng: nếu thiên về dense quá nhiều sẽ mất các match chính xác theo từ khóa, còn nếu thiên BM25 quá thì dễ bỏ lỡ ngữ nghĩa. Quan trọng nhất, retrieval tốt giúp prompt grounded hơn và giảm hallucination ở bước generation.

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn 

Điều làm tôi bất ngờ là có những câu tưởng đơn giản nhưng pipeline vẫn trả lời sai vì retrieval không lấy đúng chunk dù tài liệu có chứa đáp án. Lỗi mất nhiều thời gian nhất là hiện tượng “top-k có tài liệu đúng nhưng chunk đứng quá thấp”, khiến model sinh câu trả lời dựa trên context chưa đủ mạnh. Ban đầu tôi giả thuyết vấn đề nằm ở prompt generation, nhưng khi kiểm tra log retrieved documents thì thấy nguyên nhân chính ở indexing/chunking và cách trộn dense-bm25. Một khó khăn khác là xử lý các câu hỏi vừa cần từ khóa chính xác vừa cần hiểu ngữ cảnh; dùng một phương pháp đơn lẻ cho kết quả thiếu ổn định. Sau khi điều chỉnh hybrid weighting và kiểm tra lại từng failure case, kết quả cải thiện rõ hơn so với việc chỉ chỉnh prompt.

## 4. Phân tích một câu hỏi trong scorecard 

**Câu hỏi:** “Theo policy_refund_v4, điều kiện hoàn tiền và thời hạn xử lý là gì?”

Ở baseline, hệ thống trả lời thiếu một phần điều kiện và diễn đạt khá chung chung, nên điểm factuality/groundedness không cao. Khi xem lại pipeline, lỗi chính không nằm ở generation mà ở retrieval: BM25 bắt được tài liệu đúng nhờ từ khóa “refund_v4”, nhưng dense retrieval đôi khi kéo thêm các chunk policy liên quan nhưng không trực tiếp chứa điều khoản cần trích. Khi context bị nhiễu, model có xu hướng tổng quát hóa thay vì nêu đúng điều kiện cụ thể.

Ở variant hybrid, tôi giữ BM25 để đảm bảo match chính xác tài liệu policy_refund_v4, đồng thời dùng dense để bổ sung các câu giải thích liên quan. Sau đó, tôi giảm số chunk dense trong top context và ưu tiên chunk chứa terms quan trọng như phiên bản policy, điều kiện áp dụng, thời hạn xử lý. Kết quả là câu trả lời đầy đủ hơn, sát nguồn hơn, và giảm lỗi “nói đúng ý nhưng thiếu chi tiết bắt buộc”. Điều này cho thấy với bài toán policy QA, retrieval quality và ranking strategy tác động trực tiếp đến điểm cuối nhiều hơn việc chỉ thay đổi prompt.

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? 

Nếu có thêm thời gian, tôi sẽ thử một bước re-ranking nhẹ sau hybrid retrieval để ưu tiên chunk có overlap cao với các thực thể quan trọng (mã policy, mốc thời gian, điều kiện). Tôi cũng muốn làm ablation rõ ràng cho từng cấu hình top-k dense, top-k BM25 và trọng số fusion vì kết quả eval cho thấy một số câu vẫn dao động giữa các lần chạy. Mục tiêu là tăng tính ổn định, không chỉ tăng điểm trung bình.

---
