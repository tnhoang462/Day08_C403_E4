# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Xuân Mong
**Vai trò trong nhóm:**  Eval Owner 
**Ngày nộp:** 4/13/2026 
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> Mô tả cụ thể phần bạn đóng góp vào pipeline:
> - Sprint nào bạn chủ yếu làm?
> - Cụ thể bạn implement hoặc quyết định điều gì?
> - Công việc của bạn kết nối với phần của người khác như thế nào?

Trong Lab này, tôi chịu trách nhiệm chính trong **Sprint 4 (Evaluation & Scorecard)**. Cụ thể, tôi đã implement 2 hàm evaluation quan trọng để chấm điểm chất lượng RAG pipeline:
1. **`score_answer_relevance`**: Đo lường mức độ liên quan của câu trả lời so với câu hỏi bằng chiến lược LLM-as-Judge. Tôi đã thiết kế prompt để LLM chấm trên thang điểm 1-5, đồng thời thiết lập kết nối API với mô hình `openai/gpt-oss-120b` của NVIDIA để tự động trả về điểm số và lý do (reason).
2. **`score_context_recall`**: Đánh giá module Retrieval bằng cách tự động đối chiếu xem `expected_sources` có nằm khớp trong danh sách các retrieved chunks hay không (tính recall). Hàm này thực hiện partial/exact matching để mang lại độ chính xác ổn định mà không cần tốn chi phí gọi LLM.

Phần công việc này kết nối trực tiếp đến phần khung của nhóm bằng cách lượng hóa chất lượng các output sinh ra từ retrieval và generation. Nhờ các metric này, nhóm có căn cứ đánh giá baseline test và so sánh chính xác sự cải thiện sau A/B tuning (ví dụ: chuyển từ dense search sang hybrid + rerank).

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, thông qua việc xây dựng các hàm chấm điểm, tôi đã hiểu sâu rễ vấn đề về **Evaluation Loop** và khái niệm **LLM-as-a-Judge**:
- **Tách biệt Evaluation giữa Retrieval và Generation**: Trước đây tôi thường gộp chung chất lượng câu trả lời là một. Nhưng khi code `score_context_recall`, tôi nhận ra retrieval evaluation là bài toán "tìm có trúng đích không" (exact/partial match source documents), nó hoàn toàn có thể đo tự động và rẻ tiền bằng string matching. Ngược lại, để đánh giá generation (`score_answer_relevance`), câu chữ có thể biến hóa đa dạng nên bắt buộc phải lệ thuộc vào mức độ "hiểu ý nghĩa" của LLM.
- **LLM-as-a-Judge**: Thay vì nhờ người dùng đọc từng câu rồi điền Google Form khảo sát, tôi đã học được cách gói ghém thang rubric (1-5 điểm) vào trong một prompt để ủy quyền cho RAG Model tự đánh giá một RAG Response. Việc LLM vừa trả về "điểm" và "lý thuyết" (reasoning) đem lại sự lượng hoá minh bạch, giúp tự động hóa toàn bộ quá trình tuning vòng lặp của nhóm.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Trái với giả thuyết ban đầu là phiên bản Variant sẽ nâng điểm vượt trội, điều tôi ngạc nhiên và gặp khó khăn nhất là toàn bộ pipeline của **Variant (Hybrid + Rerank)** mắc lỗi crash hoàn toàn (`ERROR: 'NoneType' object is not iterable`), bị abort và trả về kết quả rỗng (0 hoặc N/A) cho tất cả metric điểm số, khiến việc lấy dữ liệu so sánh A/B bị gián đoạn.
Ngoài ra, đối chiếu kết quả **Baseline**, tôi rất ngạc nhiên khi đo lường thực tế trả về trung bình `Context Recall` đạt tối đa 5.00/5 — tức là bộ tìm kiếm bốc trúng hoàn hảo source cần thiết (lấy đúng), nhưng `Faithfulness` lại chỉ đạt vỏn vẹn 2.90/5. Đo đạc bằng metrics này cho thấy RAG pipeline đang mắc ảo giác (hallucination) nghèo nàn cốt truyện: dù có đủ và đúng tài liệu, LLM ở khâu generation vẫn không thèm bám sát context mà tự lồng ghép thêm rất nhiều chi tiết (ví dụ: bịa thêm mã điều khoản hay phiên bản), làm hỏng độ tin cậy của câu trả lời sinh ra.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Khi làm việc remote, tôi phải dùng VPN và được kết nối trên tối đa bao nhiêu thiết bị?" (ID: gq02 - Cross-Document)

**Phân tích:**
Thông qua bản scorecard sau khi chạy test thực tế ghi nhận vào tệp CSV, câu hỏi này là minh chứng cụ thể nhất để phân rã lỗi nằm ở bước nào trong pipeline:
- Nhìn vào bảng kết quả của **Baseline (Dense Retrieval thuần)**, metric `Context Recall` trả về 5/5, nghĩa là thuật toán tìm đã kéo trúng phóc 2/2 source pdf về VPN và thiết bị. Tuy nhiên, `Faithfulness` bị đánh tụt tàn bạo xuống 1/5 và `Completeness` chỉ đạt 2/5. Lý do là LLM chỉ nhặt đúng được giới hạn 2 thiết bị nhưng lại múa may tự chắp vá bịa đặt ra "tài khoản VPN", hoàn toàn ngó lơ các quy định bắt buộc phải dùng Cisco AnyConnect trong tài liệu (hallucinated claims). Suy ra, lỗi hoàn toàn nằm ở bước Generation (prompt generation chưa đủ "neo" cứng bắt model bám sát context), chứ không phải do khâu Indexing/Retrieval yếu kém.
- Khi chuyển sang **Variant (Hybrid + Rerank)**, kịch bản lý tưởng là prompt sẽ tốt hơn nhưng do hệ thống bị sập exception `ERROR: 'NoneType' object is not iterable` ngay quá trình chạy pipeline, thành ra quá trình không xuất ra file output trả lời nên toàn bộ Evaluation báo `Pipeline error`. Do đó, phiên bản Variant ở bước này chưa thể đo lường và hoàn toàn thất bại so với Baseline hiện tại.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Tôi sẽ tập trung cải tiến tính chính xác của cơ chế LLM-as-a-Judge. Qua khảo sát log eval thô, tôi nhận thấy Judge đôi khi vẫn còn hiện tượng "dễ dãi" do thiên kiến ảo giác ngầm định (chấm `score_answer_relevance` điểm cao 4 hay 5 mặc dù câu trả lời còn mang hàm ý doán mò tự biên).
Phương án tôi muốn thử nếu có thêm thời gian là áp dụng **Few-shot Prompting** vào prompt ở `_llm_judge`: truyền trực tiếp 1 ví dụ "thế nào mới xứng đáng 5 điểm" và 1 ví dụ "lạc đề 2 điểm" vào template prompt. Neo tham chiếu bằng few-shot này sẽ tạo thước đo tiêu chuẩn ép LLM phán xử khắt khe và như một kỹ sư chuyên gia hơn, thay vì zero-shot prompt nguyên thuỷ hơi khó control lỏng lẻo như hiện tại.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
