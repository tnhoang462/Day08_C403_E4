# AI in Action — Lecture Slides: Day 08 · 09 · 10

> **Course:** AI in Action – Phase 1 (Nền tảng)
> **Topic block:** Từ RAG đến Multi-Agent đến Data Pipeline — xây dựng hệ AI vận hành thực tế

---

## Overview

Ba ngày học này tạo thành một mạch kiến thức liên tục: cùng một artifact (trợ lý nội bộ cho khối CS + IT Helpdesk) được nâng cấp qua từng ngày — từ RAG có kiểm soát, sang điều phối đa agent, đến tầng data pipeline và observability đảm bảo hệ chạy bền.

```
Day 08 ─ RAG grounded          →   Day 09 ─ Supervisor-Workers      →   Day 10 ─ Data + Observe
   Retrieve đúng đoạn               Route + trace + MCP                  Freshness · quality · alert
```

---

## Day 08 — RAG Pipeline

**File:** [`day08/lecture-08.html`](day08/lecture-08.html)

### Vấn đề mở bài
Vector store đã có, nhưng agent vẫn trả lời sai — tại sao?

### Nội dung chính

| Chủ đề | Chi tiết |
|--------|----------|
| **Indexing pipeline** | Chunking đúng, metadata rõ, freshness có kiểm soát |
| **Retrieval strategy** | Dense vs Sparse vs Hybrid; query transformation; top-k & rerank funnel |
| **Grounded prompting** | Inject context đúng cách để model bớt hallucinate |
| **RAG evaluation** | Đo bằng scorecard (faithfulness, relevance, correctness), không bằng cảm giác |

### Hoạt động trong lớp
1. **Error Tree** — phân tích nguyên nhân gốc rễ của RAG failure
2. **Chunking Clinic** — so sánh chiến lược chunk
3. **Retrieval decision map** — chọn chiến lược phù hợp với use case
4. **Prompt Surgery** — sửa grounded prompt thực tế

### Deliverables
- `rag_pipeline.py` — indexing + retrieval end-to-end
- `eval_scorecard.csv` — kết quả đo retrieval và answer quality
- `rag_architecture.md` — tài liệu thiết kế

---

## Day 09 — Multi-Agent & Kết Nối Hệ Thống

**File:** [`day09/lecture-09.html`](day09/lecture-09.html)

### Vấn đề mở bài
Một agent giỏi vẫn quá tải khi bài toán phức tạp — khi nào nên tách hệ?

### Nội dung chính

| Chủ đề | Chi tiết |
|--------|----------|
| **Multi-agent patterns** | Supervisor-Workers, Pipeline, Peer-to-peer, Hierarchical |
| **Supervisor-Worker** | Route theo tín hiệu quan sát được (task type, confidence, risk) |
| **Worker contract** | Rõ input · rõ output · rõ lỗi — chuẩn để test và thay thế |
| **MCP architecture** | Agent cắm vào năng lực bên ngoài (tool, API) theo chuẩn chung |
| **A2A vs MCP** | Phân biệt giao việc cho agent khác vs lấy capability từ bên ngoài |
| **LangGraph** | Node, edge, state, route function, HITL checkpoint |
| **Trace & Observability** | Ghi task · route reason · worker IO · answer cuối để debug và học |

### Hoạt động trong lớp
1. **Agent overload map** — xác định điểm single-agent quá tải
2. **Chọn pattern** — so sánh 4 pattern cho 3 use case thực tế
3. **Tách artifact Day 08** — chia RAG agent thành supervisor + workers
4. **Phân biệt MCP với A2A** — hands-on classification

### Deliverables
- `supervisor_agent.py` — router với route logic rõ ràng
- `worker_*.py` — retrieval, policy, synthesis workers
- `trace_log.jsonl` — trace ghi đủ bước cho mỗi run
- `mcp_config.json` — khai báo tool/MCP connection

---

## Day 10 — Data Pipeline & Data Observability

**File:** [`day10/lecture-10.html`](day10/lecture-10.html)

### Vấn đề mở bài
Data từ database công ty đột nhiên sai — agent hallucinate. Hệ của bạn có biết không?

> *Garbage in → garbage out. Đừng debug model trước khi debug pipeline.*

### Nội dung chính

| Chủ đề | Chi tiết |
|--------|----------|
| **ETL vs ELT** | Transform trước hay sau load — quyết định dựa trên governance, latency, cost |
| **Batch vs Streaming** | Trade-off latency vs complexity; khi nào cần streaming thực sự |
| **Ingestion layer** | CDC, rate limit, backpressure, retry + backoff, DLQ |
| **Data transformation** | Làm sạch PII, chuẩn hoá schema, dedupe, encoding |
| **Data quality as code** | Expectation suite chạy như unit test trong CI/CD |
| **5 pillars of observability** | Freshness · Volume · Distribution · Schema · Lineage |
| **Orchestration** | DAG, dependency, retry, idempotency, SLA alert |
| **Incident triage** | Runbook — detect → isolate → fix → verify → post-mortem |

### Hoạt động trong lớp
1. **ETL hay ELT? Batch hay Streaming?** — phân loại 4 tình huống thực tế
2. **Source map & failure points** — ingestion plan cho DB + API + PDF
3. **Dirty data repair** — sửa dataset có missing, duplicate, wrong format
4. **Incident triage** — xử lý freshness breach theo runbook

### Deliverables
- `etl_pipeline.py` — ingest → clean → validate → embed end-to-end
- `quality/expectations.py` — expectation suite kiểm tra data quality
- `monitoring/freshness_check.py` — freshness + volume monitor
- `before_after_eval.csv` — bằng chứng data quality ảnh hưởng answer quality
- `pipeline_architecture.md` + `data_contract.md` + `runbook.md`

---

## Mạch xuyên suốt 3 ngày

```
                  ┌──────────────────────────────────────────────────────┐
                  │              Trợ lý nội bộ CS + IT Helpdesk          │
                  └──────────────────────────────────────────────────────┘
                            ↑               ↑               ↑
                         Day 08          Day 09          Day 10
                      RAG grounded   Supervisor +    Data pipeline
                      retrieve đúng  workers route   + observability
                      đoạn, đo được  trace rõ,MCP    detect issue
                                     A2A                 sớm
```

Mỗi ngày xây trên artifact của ngày trước:
- **Day 08** cung cấp `rag_pipeline` làm nền retrieval
- **Day 09** bọc nó vào `retrieval_worker`, thêm `policy_worker`, `synthesis_worker` và supervisor route
- **Day 10** đảm bảo dữ liệu feeding vào toàn bộ hệ không bị stale, dirty hay missing

---

## Tech Stack

| Layer | Công cụ |
|-------|---------|
| LLM | Claude / GPT-4o |
| Embedding | text-embedding-3-small / BGE |
| Vector store | ChromaDB / Qdrant / pgvector |
| Orchestration | LangGraph / CrewAI |
| MCP | MCP SDK |
| Pipeline | Python ETL / Prefect / Airflow |
| Quality | Great Expectations |
| Monitoring | Custom + Grafana |

---

## How to Use

Mỗi slide deck là một file HTML độc lập, chạy thẳng trên browser — không cần cài thêm server:

```bash
# Mở trực tiếp trong browser
open day08/lecture-08.html
open day09/lecture-09.html
open day10/lecture-10.html
```

**Điều hướng:**
- `←` / `→` hoặc `Space` — chuyển slide
- `N` — toggle speaker notes
- `Home` / `End` — về đầu / cuối

---

*AI in Action · VinUniversity · 2026*
