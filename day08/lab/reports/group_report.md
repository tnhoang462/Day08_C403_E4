# Báo Cáo Nhóm — Lab Day 08: Full RAG Pipeline

**Tên nhóm:** C401 
**Thành viên:**
| Tên | Vai trò | Email / ID |
|-----|---------|-------|
| Nguyễn Ngọc Thắng | Tech Lead & RAG Pipeline Owner | 2A202600191 |
| Trần Nhật Hoàng | Retrieval Owner / Debug Support | 2A202600431 |
| Trương Đặng Gia Huy | Eval Owner | 2A202600436 |
| Lê Quý Công | Indexing Owner & Unit Test | 2A202600104 |
| Phạm Đỗ Ngọc Minh | Tech Lead, Eval Owner & Documentation | 2A202600256 |
| Trần Minh Toàn | Retrieval Owner | 2A202600297 |
| Nguyễn Xuân Mong | Eval Owner |2A202600246 |

**Ngày nộp:** 14/4/2026  
**Repo:** https://github.com/tnhoang462/Day08_C403_E4  
**Độ dài khuyến nghị:** 600–900 từ

---

## 1. Pipeline nhóm đã xây dựng (150–200 từ)

> Mô tả ngắn gọn pipeline của nhóm:
> - Chunking strategy: size, overlap, phương pháp tách (by paragraph, by section, v.v.)
> - Embedding model đã dùng
> - Retrieval mode: dense / hybrid / rerank (Sprint 3 variant)

**Chunking decision:**
Nhóm sử dụng `CHUNK_SIZE = 400` tokens, `CHUNK_OVERLAP = 80` tokens. Phương pháp cắt kết hợp giữa cắt theo cấu trúc văn bản (Section/Heading `=== ... ===`) trước, sau đó nếu phần section quá lớn mới dùng RegExp để dự phòng chẻ nhỏ thêm theo đoạn văn (paragraph via `\n\n`) nhằm đảm bảo ngữ cảnh của một điều khoản hay khái niệm không bị cắt đứt gãy. Metadata bổ sung được lấy chi tiết từ header (`source`, `department`, `effective_date`, `access`) để ghim chặt vào từng chunk.

_________________

**Embedding model:**
Sử dụng mô hình OpenAI `text-embedding-3-small` qua API do có độ sâu chiều (1536 dims) và năng lực biểu diễn ngữ nghĩa vượt trội. Pipeline cũng được thiết kế fallback dùng `paraphrase-multilingual-MiniLM-L12-v2` của `sentence-transformers` khi chạy local không API key.

_________________

**Retrieval variant (Sprint 3):**
Nhóm chọn sử dụng **Hybrid Retrieval kết hợp Cross-Encoder Rerank**.
Lý do: Lối tiếp cận cũ Dense Search lấy ngữ nghĩa tốt nhưng thường xuyên bỏ sót exact match các mã lỗi hay ký hiệu (như `SLA P1` hay `ERR-403`). Hybrid dung hòa giữa Dense và Sparse (BM25) qua thuật toán Reciprocal Rank Fusion (RRF), sau đó tận dụng Rerank (VD: `ms-marco`) chấm điểm để chọn lại các chunk gãy góc và chính xác nhất đẩy lên top.

_________________

---

## 2. Quyết định kỹ thuật quan trọng nhất (200–250 từ)

> Chọn **1 quyết định thiết kế** mà nhóm thảo luận và đánh đổi nhiều nhất trong lab.
> Phải có: (a) vấn đề gặp phải, (b) các phương án cân nhắc, (c) lý do chọn.

**Quyết định:** Chế ngự tình trạng Hallucination (Ảo giác) bằng Grounded Prompting kết hợp Hybrid Retrieval.

**Bối cảnh vấn đề:**
Ở phiên bản baseline với Dense Retrieval và System Prompt mở, mô hình liên tục tự sáng tác thông tin. Đối với các dữ kiện chưa đủ ngữ cảnh hoặc hoàn toàn không nằm trong kho tài liệu, LLM vì thói quen cố gắng trả lời đã tự lấy knowledge bên ngoài (như câu `gq09` tự suy luận đổi mật khẩu mất 90 ngày) hoặc tự biên tự diễn án phạt khi vi phạm `SLA P1` (câu `gq07`) khiến cho điểm Faithfulness chạm đáy thảm hại thấp nhất là 1/5 và trung bình chỉ đạt 2.60/5.

**Các phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| **Chỉ dùng Baseline Dense** | Dễ Code, chi phí gọi API truy xuất thấp nhất | Lấy thiếu các từ khóa chuyên ngành, Hallucination (mô hình bịa đặt cao), Completeness thấp |
| **Đổi mô hình LLM lớn hơn** | Cải thiện hành văn tự nhiên | Tốn chi phí, tốn thời gian tuning, ảo giác chưa chắc được giải quyết gốc rễ |
| **Sửa Prompting + Hybrid Rerank** | Xử lý triệt để Keyword miss, triệt tiêu ảo giác, buộc LLM phải Abstain (Từ chối) khi không đủ ngữ cảnh | Luồng pipeline phức tạp hơn, có thể bị lỗi Code tích hợp (crashed NoneType), tăng latency nhẹ |

**Phương án đã chọn và lý do:**
Nhóm lựa chọn phương án **Sửa Prompting + Hybrid Rerank**. Grounded Prompt chứa các luật "Evidence-Only" rất khắt khe bắt buộc model trích dẫn `[1] doc | section`, kèm mệnh lệnh "từ chối từ chối trả lời nếu ngữ cảnh không đủ mật thiết". Đi kèm đó là bù đắp BM25 vào Retrieval để vá các lỗ hổng tìm kiếm từ khóa. Quyết định tuy khó ở khâu merge luồng code nhưng đây là cốt lõi để xây một hệ thống tra cứu tra soát an toàn và đáng tin cậy.

**Bằng chứng từ scorecard/tuning-log:**
Trong quá trình triển khai Variant 1 (ghi chú ở `tuning-log.md`), Faithfulness Score có pha bứt tốc nhảy từ 2.60/5 lên 4.80/5, Answer Relevance từ 4.20/5 lên 4.90/5 và Completeness chạm đạt 4.50/5.

---

## 3. Kết quả grading questions (100–150 từ)

> Sau khi chạy pipeline với grading_questions.json (public lúc 17:00):
> - Câu nào pipeline xử lý tốt nhất? Tại sao?
> - Câu nào pipeline fail? Root cause ở đâu (indexing / retrieval / generation)?
> - Câu gq07 (abstain) — pipeline xử lý thế nào?

**Ước tính điểm raw:** 85-90 / 98 (Dựa trên score trung bình các metric ~4.5/5 tại bản Variant 1).

**Câu tốt nhất:** Nhóm xử lý tốt các câu dạng Single Document trích dẫn từ khóa chuẩn. Đồng thời cực kì ấn tượng ở Case **SLA P1 (gq07)** đã có thể xử lý mượt mà. Lý do: Mô hình bám theo luật trong Prompt, biết cách dừng lại và Abstain nếu query yêu cầu chi tiết không tồn tại.

**Câu fail:** Các câu dạng Cross-Document (Ví dụ **gq02** - Tổng hợp thiết bị và VPN, hay **gq03** rules refund). Root cause: **Generation**. LLM gặp trục trặc "lười biếng" chỉ dừng lại ở việc đọc 1 source, không cố lấy tổng hợp từ 2 source dù cho `Context Recall` đã đẩy về đủ 2 chunks tài liệu.

**Câu gq07 (abstain):** Ở Baseline, LLM tự lấy pre-processing knowledge bịa ra số tiền phạt. Ở Variant, Pipeline trả lời đúng mẫu chuẩn của người thiết lập: *"Tôi không tìm thấy thông tin này trong tài liệu nội bộ hiện có."*

---

## 4. A/B Comparison — Baseline vs Variant (150–200 từ)

> Dựa vào `docs/tuning-log.md`. Tóm tắt kết quả A/B thực tế của nhóm.

**Biến đã thay đổi (chỉ 1 biến):** Cập nhật Prompt (Grounded Strict Constraints) và chuyển Retrieval sang Hybrid Mode (BM25 + Dense) kết hợp Rerank.

| Metric | Baseline | Variant 1 | Delta |
|--------|---------|---------|-------|
| Faithfulness | 2.60/5 | 4.80/5 | +2.20 |
| Answer Relevance | 4.20/5 | 4.90/5 | +0.70 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.40/5 | 4.50/5 | +1.10 |

**Kết luận:**
Variant thể hiện tính ưu việt rõ ràng so với Baseline. Điểm nhấn lớn nhất là `Faithfulness` được kéo tăng +2.20. Nhờ Hybrid Rerank kết hợp Keyword Extraction trực quan từ văn bản vào prompt, mọi ý đồ hallucination tự suy diễn gần như tuyệt diệt. Pipeline giờ đây phản hồi chân thực theo data đưa vào, và `Completeness` (+1.10) cho thấy thông tin trích dẫn không chỉ thật mà còn đầy đủ và bao quát hơn.

---

## 5. Phân công và đánh giá nhóm (100–150 từ)

> Đánh giá trung thực về quá trình làm việc nhóm.

**Phân công thực tế:**

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Lê Quý Công | Chunking & Data Indexing, Unit Test, Fake Config | Sprint 1 |
| Nguyễn Ngọc Thắng | Generation, Context Builder, Architecture Design (Tech Lead) | Sprint 2, 3 |
| Trần Nhật Hoàng | Retrieve_hybrid và Rerank, Architecture Design | Sprint 2, 3 |
| Trần Minh Toàn | Baseline vs Hybrid Retrieval Comparison | Sprint 1, 3 |
| Phạm Đỗ Ngọc Minh | LLM-as-a-Judge Eval Loop, Documentation Owner | Sprint 4 |
| Nguyễn Xuân Mong | thiết kế score_answer_relevance và score_context_recall | Sprint 4 |
| Trương Đặng Gia Huy | Vắng mặt do đi học, đọc lại code và kết quả sau lab để tự học | - |

**Điều nhóm làm tốt:**
Tuy quy trình thiết kế Pipeline phức tạp, đặc biệt là bước biến đổi từ vòng đời cũ (Dense) sang vòng đời mới (Hybrid có Rerank), các bộ phận Eval đã làm rất tốt việc đóng gói các metrics chấm điểm như `Faithfulness`, `Answer Relevance`. Qua Evaluation độc lập này, nhóm sớm tự bắt lỗi Hallucination thay vì đợi giáo viên kiểm tra thủ công.

**Điều nhóm làm chưa tốt:**
Quá trình ghép nhánh (merge code) từ script `retrieval.py` sang hàm tổng `rag_answer.py` xử lý còn yếu, dẫn đến đôi lúc Variant bị crash code (`NoneType object is not iterable`) cản trở pipeline test hoàn thiện tự động và đồng bộ.

---

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì? (50–100 từ)

> 1–2 cải tiến cụ thể với lý do có bằng chứng từ scorecard.

1. **Self-Query Retriever & Metadata Filtering:** Phân tích thấy `effective_date` thường gây lúng túng khi policy update. Nhóm sẽ chèn LLM tự động bắt lấy yếu tố ngày tháng năm (vd: yêu cầu update rules năm 2026/ v4) để tự filter database Chroma trước khi search.
2. **Decomposition (Phân rã đa nhánh):** Điểm `Completeness` vẫn gãy xước ở các câu Cross-Document (gq02, gq03). Nhóm sẽ cài script tách Query dài phức tạp ra làm 2 Query ngắn chạy độc lập, rồi mới đưa LLM tổng hợp Context cuối để hết tình trạng "trả lời nửa vời" do context loãng.

---

*File này lưu tại: `reports/group_report.md`*  
*Commit sau 18:00 được phép theo SCORING.md*
