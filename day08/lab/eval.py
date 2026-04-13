"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer, call_llm

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3) — Hybrid + Rerank
# Lý do: corpus có cả câu tự nhiên (policy) lẫn mã lỗi, tên riêng (P1, ERR-403, SLA).
# Hybrid giữ được exact keyword recall; rerank loại bỏ noise sau search rộng.
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid_rerank",
}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def _llm_judge(prompt: str, fallback_score: int = 3) -> Dict[str, Any]:
    """
    Gọi LLM để chấm điểm và trả về {"score": int, "reason": str}.
    Nếu LLM không khả dụng hoặc output không parse được, trả về fallback.
    """
    try:
        raw = call_llm(prompt)
        # Extract JSON from response
        import re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", fallback_score))
            reason = str(data.get("reason", ""))
            return {"score": max(1, min(5, score)), "notes": reason}
        else:
            # Try to extract just a number
            num_match = re.search(r'\b([1-5])\b', raw)
            if num_match:
                return {"score": int(num_match.group(1)), "notes": raw.strip()[:200]}
    except Exception as e:
        print(f"[_llm_judge] Lỗi: {e}")

    return {"score": fallback_score, "notes": "LLM judge unavailable — default score used"}


def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    Câu hỏi: Model có tự bịa thêm thông tin ngoài retrieved context không?

    Thang điểm 1-5:
      5: Mọi thông tin trong answer đều có trong retrieved chunks
      4: Gần như hoàn toàn grounded, 1 chi tiết nhỏ chưa chắc chắn
      3: Phần lớn grounded, một số thông tin có thể từ model knowledge
      2: Nhiều thông tin không có trong retrieved chunks
      1: Câu trả lời không grounded, phần lớn là model bịa

    Sử dụng LLM-as-Judge.
    """
    if not chunks_used or not answer or answer.startswith("PIPELINE_NOT_IMPLEMENTED") or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error or no chunks — skip scoring"}

    context_preview = "\n---\n".join(
        c["text"][:300] for c in chunks_used
    )

    prompt = f"""You are an evaluation judge for a RAG system.
Given the retrieved context and the model's answer, rate the FAITHFULNESS of the answer.

Faithfulness measures whether ALL claims in the answer are directly supported by the retrieved context.
A faithful answer does NOT introduce information beyond what is in the context.

Scoring scale (1-5):
  5 = Every claim in the answer is explicitly supported by the context
  4 = Almost fully grounded, one minor detail is uncertain
  3 = Mostly grounded, but some information may come from model's prior knowledge
  2 = Several claims are not supported by the context
  1 = Answer is largely hallucinated or not grounded in the context

Retrieved context:
{context_preview}

Model answer:
{answer}

Output ONLY a JSON object with two keys: "score" (integer 1-5) and "reason" (one sentence).
Example: {{"score": 4, "reason": "Answer is mostly grounded but adds one unsupported detail."}}"""

    return _llm_judge(prompt, fallback_score=3)


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    Câu hỏi: Model có bị lạc đề hay trả lời đúng vấn đề cốt lõi không?

    Thang điểm 1-5:
      5: Answer trả lời trực tiếp và đầy đủ câu hỏi
      4: Trả lời đúng nhưng thiếu vài chi tiết phụ
      3: Trả lời có liên quan nhưng chưa đúng trọng tâm
      2: Trả lời lạc đề một phần
      1: Không trả lời câu hỏi

    Sử dụng LLM-as-Judge.
    """
    if not answer or answer.startswith("PIPELINE_NOT_IMPLEMENTED") or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error — skip scoring"}

    prompt = f"""You are an evaluation judge for a RAG system.
        Rate how well the model's answer addresses the user's question (ANSWER RELEVANCE).

        Scoring scale (1-5):
        5 = Directly and completely answers the question
        4 = Answers correctly but misses minor secondary details
        3 = Related but does not fully address the core of the question
        2 = Partially off-topic
        1 = Does not answer the question at all

        User question: {query}

        Model answer: {answer}

        Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt, fallback_score=3)


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),  # Convert to 1-5 scale
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?
    Câu hỏi: Answer có bao phủ đủ thông tin so với expected_answer không?

    Thang điểm 1-5:
      5: Answer bao gồm đủ tất cả điểm quan trọng trong expected_answer
      4: Thiếu 1 chi tiết nhỏ
      3: Thiếu một số thông tin quan trọng
      2: Thiếu nhiều thông tin quan trọng
      1: Thiếu phần lớn nội dung cốt lõi

    Sử dụng LLM-as-Judge.
    """
    if not answer or answer.startswith("PIPELINE_NOT_IMPLEMENTED") or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error — skip scoring"}

    if not expected_answer:
        return {"score": None, "notes": "No expected answer provided"}

    prompt = f"""You are an evaluation judge for a RAG system.
Rate the COMPLETENESS of the model's answer by comparing it to the reference (expected) answer.

Completeness measures whether the model covers all the key points in the reference answer.

Scoring scale (1-5):
  5 = All key points from the expected answer are present
  4 = Missing one minor detail
  3 = Missing some important information
  2 = Missing much of the important information
  1 = Missing most of the core content

User question: {query}

Expected answer (reference): {expected_answer}

Model answer: {answer}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt, fallback_score=3)


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id    = q["id"]
        query          = q["question"]
        expected_answer  = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category         = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer     = result["answer"]
            chunks_used = result["chunks_used"]

        except Exception as e:
            answer      = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith    = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall   = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id":                   question_id,
            "category":             category,
            "query":                query,
            "answer":               answer,
            "expected_answer":      expected_answer,
            "faithfulness":         faith["score"],
            "faithfulness_notes":   faith["notes"],
            "relevance":            relevance["score"],
            "relevance_notes":      relevance["notes"],
            "context_recall":       recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness":         complete["score"],
            "completeness_notes":   complete["notes"],
            "config_label":         label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:120]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    print(f"\n{'—'*40}")
    print(f"Summary [{label}]:")
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"  Average {metric}: {avg:.2f}/5" if avg else f"  Average {metric}: N/A")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results  if r[metric] is not None]

        b_avg  = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg  = sum(v_scores) / len(v_scores) if v_scores else None
        delta  = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'ID':<6} {'BL F/R/Rc/C':<22} {'VR F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid   = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([str(b_row.get(m, "?")) for m in metrics])
        v_scores_str = "/".join([str(v_row.get(m,  "?")) for m in metrics])

        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better  = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """Tạo báo cáo tóm tắt scorecard dạng markdown."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg is not None else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        note = r.get("faithfulness_notes", "")
        if note:
            note = note[:60]
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {note} |\n")

    md += "\n## Answers\n\n"
    for r in results:
        md += f"### [{r['id']}] {r['query']}\n"
        md += f"**Answer:** {r['answer']}\n\n"
        md += f"**Expected:** {r['expected_answer']}\n\n"
        md += "---\n\n"

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")
    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Chạy Baseline (Sprint 2) ---
    print("\n--- Chạy Baseline (Dense) ---")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")

    except Exception as e:
        print(f"Baseline lỗi: {e}")
        baseline_results = []

    # --- Chạy Variant (Sprint 3) ---
    print("\n--- Chạy Variant (Hybrid + Rerank) ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard variant lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")

    except Exception as e:
        print(f"Variant lỗi: {e}")
        variant_results = []

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv",
        )

    print("\n✓ Sprint 4 hoàn thành!")
    print(f"  Xem kết quả tại: {RESULTS_DIR}")