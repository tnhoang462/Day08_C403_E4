"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =============================================================================
# SHARED — Cached ChromaDB client + collection
# Tránh mở nhiều PersistentClient cùng lúc (gây locking issue)
# =============================================================================

_chroma_collection = None


def _get_collection():
    """Trả về ChromaDB collection đã cache. Khởi tạo 1 lần duy nhất."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        from index import CHROMA_DB_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _chroma_collection = client.get_collection("rag_lab")
    return _chroma_collection


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(
    query: str,
    top_k: int = TOP_K_SEARCH,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về
        where: Optional ChromaDB metadata filter, ví dụ:
               {"department": "Finance"}
               {"access": "public"}
               {"$and": [{"department": "HR"}, {"access": "internal"}]}

    Returns:
        List các dict, mỗi dict chứa "id", "text", "metadata", "score"
    """
    from index import get_embedding

    collection = _get_collection()
    query_embedding = get_embedding(query)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance = 1 - cosine_similarity
        score = 1.0 - dist
        chunks.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "score": score,
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

_bm25_cache = None


def _build_bm25_index():
    """
    Load tất cả chunks từ ChromaDB và tạo BM25 index.
    Cache lại để không rebuild mỗi lần gọi.
    """
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    from rank_bm25 import BM25Okapi

    collection = _get_collection()

    # Lấy toàn bộ documents từ collection (kèm IDs để truy vết)
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    # Tokenize: lowercase + split by whitespace
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_cache = (bm25, all_ids, all_docs, all_metas)
    return _bm25_cache


def retrieve_sparse(
    query: str,
    top_k: int = TOP_K_SEARCH,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về
        where: Optional metadata filter (áp dụng post-hoc vì BM25 không hỗ trợ native)
    """
    bm25, all_ids, all_docs, all_metas = _build_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy nhiều hơn top_k nếu có filter, vì sẽ loại bớt sau
    fetch_k = top_k * 3 if where else top_k
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:fetch_k]

    chunks = []
    for idx in top_indices:
        meta = all_metas[idx]

        # Post-hoc metadata filter
        if where and not _matches_where(meta, where):
            continue

        chunks.append({
            "id": all_ids[idx],
            "text": all_docs[idx],
            "metadata": meta,
            "score": float(scores[idx]),
        })

        if len(chunks) >= top_k:
            break

    return chunks


def _matches_where(meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """
    Kiểm tra metadata có match điều kiện where filter không.
    Hỗ trợ simple key-value match và $and/$or operators (tương thích ChromaDB syntax).
    """
    if "$and" in where:
        return all(_matches_where(meta, cond) for cond in where["$and"])
    if "$or" in where:
        return any(_matches_where(meta, cond) for cond in where["$or"])

    for key, value in where.items():
        if key.startswith("$"):
            continue
        if meta.get(key) != value:
            return False
    return True


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = dense_weight * 1/(K + rank_dense) + sparse_weight * 1/(K + rank_sparse)
    """
    RRF_K = 60  # Hằng số RRF tiêu chuẩn

    dense_results = retrieve_dense(query, top_k, where=where)
    sparse_results = retrieve_sparse(query, top_k, where=where)

    # Dùng chunk ID làm key (ổn định hơn text — text có thể trùng do overlap)
    chunk_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk["id"]
        chunk_map[key] = chunk
        rrf_scores[key] = dense_weight * (1.0 / (RRF_K + rank + 1))

    for rank, chunk in enumerate(sparse_results):
        key = chunk["id"]
        if key not in chunk_map:
            chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight * (1.0 / (RRF_K + rank + 1))

    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    results = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = rrf_scores[key]
        results.append(chunk)

    return results


# =============================================================================
# RERANK (Cross-encoder — cached)
# =============================================================================

_cross_encoder = None


def _get_cross_encoder():
    """Cache cross-encoder model (giống cách index.py cache _st_model)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Funnel logic: Search rộng (top-10) → Rerank → Select (top-3)
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for chunk, ce_score in ranked[:top_k]:
        chunk = chunk.copy()
        chunk["rerank_score"] = float(ce_score)
        results.append(chunk)

    return results


# =============================================================================
# QUERY TRANSFORMATION
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    Yêu cầu OPENAI_API_KEY (dùng LLM để transform).
    Nếu không có key, trả về query gốc.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[transform_query] Không có OPENAI_API_KEY, trả về query gốc")
        return [query]

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    if strategy == "expansion":
        prompt = (
            f"Given the user query: '{query}'\n"
            "Generate 2-3 alternative phrasings or related search terms "
            "(in the same language as the query). Include the original query as the first item.\n"
            'Output ONLY a JSON array of strings, e.g. ["original", "alt1", "alt2"]'
        )
    elif strategy == "decomposition":
        prompt = (
            f"Break down this complex query into 2-3 simpler, self-contained sub-queries "
            f"that together cover the original intent: '{query}'\n"
            'Output ONLY a JSON array of strings, e.g. ["sub1", "sub2"]'
        )
    elif strategy == "hyde":
        prompt = (
            f"Write a short paragraph (3-5 sentences) that would be a perfect answer "
            f"to this question. Write in the same language as the query.\n"
            f"Question: '{query}'\n"
            "Output ONLY the paragraph, no explanation."
        )
    else:
        return [query]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        if strategy == "hyde":
            return [query, content]

        # expansion / decomposition → parse JSON array
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]

        queries = json.loads(content)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            if query not in queries:
                queries.insert(0, query)
            return queries

    except Exception as e:
        print(f"[transform_query] Error: {e}, fallback về query gốc")

    return [query]

 
# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================
 
def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.
 
    Format: structured snippets với source, section, score.
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    Bao gồm metadata từ index.py: source, section, department, effective_date.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        department = meta.get("department", "")
        effective_date = meta.get("effective_date", "")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
        text = chunk.get("text", "")
 
        # Header: [1] source | section | dept | date | score
        header = f"[{i}] {source}"
        if section:
            header += f" | Section: {section}"
        if department and department != "unknown":
            header += f" | Dept: {department}"
        if effective_date and effective_date != "unknown":
            header += f" | Effective: {effective_date}"
        if score > 0:
            header += f" | score={score:.2f}"
 
        context_parts.append(f"{header}\n{text}")
 
    return "\n\n".join(context_parts)
 
 
def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.
 
Question: {query}
 
Context:
{context_block}
 
Answer:"""
    return prompt
 
 
def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.
    Tự động chọn backend dựa trên API key có sẵn:
      - OPENAI_API_KEY  → OpenAI (gpt-4o-mini hoặc LLM_MODEL)
      - GOOGLE_API_KEY  → Google Gemini
 
    Dùng temperature=0 để output ổn định, dễ đánh giá.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
 
    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content
 
    elif google_key:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=512,
            ),
        )
        return response.text
 
    else:
        raise RuntimeError(
            "Không tìm thấy API key. Đặt OPENAI_API_KEY hoặc GOOGLE_API_KEY trong .env"
        )
 
 
def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    use_query_transform: bool = False,
    query_transform_strategy: str = "expansion",
    where: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → (transform) → retrieve → (rerank) → generate.
 
    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        use_query_transform: Có dùng query transformation không (Sprint 3)
        query_transform_strategy: "expansion" | "decomposition" | "hyde"
        where: Optional metadata filter (department, access, ...)
        verbose: In thêm thông tin debug
 
    Returns:
        Dict với "answer", "sources", "chunks_used", "query", "config"
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "use_query_transform": use_query_transform,
        "query_transform_strategy": query_transform_strategy,
        "where": where,
    }
 
    # --- Bước 0: Query Transformation (optional, Sprint 3) ---
    queries = [query]
    if use_query_transform:
        queries = transform_query(query, strategy=query_transform_strategy)
        if verbose:
            print(f"[RAG] Transformed queries: {queries}")
 
    # --- Bước 1: Retrieve (chạy cho mỗi query variant, dedup theo ID) ---
    retrieval_fn = {
        "dense": retrieve_dense,
        "sparse": retrieve_sparse,
        "hybrid": retrieve_hybrid,
    }.get(retrieval_mode)
 
    if retrieval_fn is None:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")
 
    seen_ids = set()
    candidates = []
    for q in queries:
        results = retrieval_fn(q, top_k=top_k_search, where=where)
        for chunk in results:
            chunk_id = chunk.get("id", chunk.get("text", "")[:80])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                candidates.append(chunk)
 
    # Sort toàn bộ candidates theo score giảm dần (best score từ bất kỳ query nào)
    score_key = "rrf_score" if retrieval_mode == "hybrid" else "score"
    candidates.sort(key=lambda c: c.get(score_key, 0), reverse=True)
 
    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} unique candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] score={c.get(score_key, 0):.3f} | {c['metadata'].get('source', '?')}")
 
    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
        if verbose:
            print(f"[RAG] After rerank: {len(candidates)} chunks")
            for i, c in enumerate(candidates):
                print(f"  [{i+1}] rerank={c.get('rerank_score', 0):.3f} | {c['metadata'].get('source', '?')}")
    else:
        candidates = candidates[:top_k_select]
 
    if verbose:
        print(f"[RAG] Final selection: {len(candidates)} chunks")
 
    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)
 
    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")
 
    # --- Bước 4: Generate ---
    answer = call_llm(prompt)
 
    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })
 
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Uncomment sau khi Sprint 3 hoàn thành:
    # print("\n--- Sprint 3: So sánh strategies ---")
    # compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    # compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")
