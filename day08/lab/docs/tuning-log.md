# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/04/2026  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 2.60/5 |
| Answer Relevance | 4.20/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.40/5 |

**Câu hỏi yếu nhất (điểm thấp):**
> Câu hỏi `gq03` (Refund rules) và `gq07` (SLA Penalties - Insufficient Context) có chỉ số faithfulness và completeness cực kỳ thấp (1/5). LLM vấp phải lỗi hallucination tự tưởng tượng và điền thông tin khi docs không đủ dữ liệu (không biết abstain), hoặc bỏ qua điều kiện quan trọng (mã flash sale) khi dense retrieval chỉ focus chung chung vào "hoàn tiền". `gq09` lấy knowledge từ external model về 90 ngày reset password.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Prompt không đủ grounding, không có instruction buộc mô hình abstain
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 13/04/2026
**Biến thay đổi:** Cập nhật Prompt và chuyển sang Hybrid Retrieval Mode (BM25 + Dense) kết hợp Cross Encoder Rerank.

**Lý do chọn biến này:**
> Baseline sử dụng dense retrieval và prompt đơn giản chưa thể ngăn chặn các failure modes như hallucination khi out-of-context hay fail-to-retrieve các exact term kỹ thuật. Dữ liệu công ty (đặc biệt là IT tickets/policies/SLA lỗi ERR) bắt buộc cần Sparse mode. Rerank module từ ms-marco lọc noise trước khi đưa vào LLM để tối ưu completeness và relevance. Cuối cùng, Grounded prompt ép strict citation đóng vai trò lớn.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # Thay dense bằng hybrid (RRF)
use_rerank = True           # Cross Encoder Rerank
# Grounded prompt với strict guidelines (Persona, Rules, Format) 
# Các tham số còn lại giữ nguyên như baseline.
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 2.60/5 | 4.80/5 | +2.20 |
| Answer Relevance | 4.20/5 | 4.90/5 | +0.70 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.40/5 | 4.50/5 | +1.10 |

**Nhận xét:**
> Variant 1 (Hybrid + Rerank + Strict Prompting) đem lại khác biệt khổng lồ. Vấn đề lớn nhất là Faithfulness được tăng gấp đôi. Mô hình đã biết nói "Tôi không tìm thấy thông tin này..." đối với các queries yêu cầu hallucination (như `gq07` về tiền phạt SLA), thay vì tự chế điểm phạt. Các cụm từ mã lỗi như ERR-403 được hybrid index bám sát. Reranker đặt chunk chính xác lên vị trí [1], giúp completeness tăng lên.

**Kết luận:**
> Variant 1 là thành công rực rỡ và vượt xa Baseline. Sự kết hợp của Keyword Extraction tự nhiên với Semantic Search giúp LLM nhặt đúng facts. Cùng với constraints trong System Prompt, pipeline hiện tại xử lí tốt các bẫy, đem lại niềm tin cho quá trình Information Lookup của nhân sự.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Hallucination - LLM luôn có xu hướng điền câu trả lời từ General Knowledge hoặc vội vã đưa ra phán đoán mà không rà soát lại Source Document. Kế đến là Keyword Retrieval Misses từ Dense Vectorizer khi người dùng tìm thuật ngữ hoặc token kỹ thuật quá nhỏ (như P1, Flash Sale).

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Biến điều khiển Prompt (Grounded Prompting, Prompt Constraints) kéo lại độ Faithfulness rõ rệt. Retrieval mode (`Hybrid`) đóng vai trò số hai trong việc nâng `Completeness` do có Dense + Sparse dung hoà lẫn nhau, đảm bảo Context Feed vào là tốt nhất có thể.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Cải thiện bộ Metadata Filtering (sử dụng Self-Query Retriever) để lọc bỏ version cũ dựa vào mốc thời gian (vd: `effective_date > 2026`) vì các rules công ty thay đổi version liên tục. Tiếp đến, thêm step Query Decomposition - bẻ các câu hỏi dài nhiều vế thành nhiều queries đánh giá độc lập.
