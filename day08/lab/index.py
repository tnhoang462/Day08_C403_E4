"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                continue
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)

    # Normalize: max 2 dòng trống liên tiếp, strip trailing whitespace per line
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+$", "", cleaned_text, flags=re.MULTILINE)
    # Chuẩn hóa dấu ngoặc kép kiểu fancy thành dấu thẳng
    cleaned_text = cleaned_text.replace("\u201c", '"').replace("\u201d", '"')
    cleaned_text = cleaned_text.replace("\u2018", "'").replace("\u2019", "'")

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.
    Ưu tiên cắt theo section heading (=== ... ===), sau đó split theo paragraph nếu quá dài.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Split theo heading pattern "=== ... ==="
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Split text dài thành chunks với overlap.
    Ưu tiên cắt theo paragraph (\n\n), rồi mới fallback cắt theo câu/ký tự.
    """
    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    # Split theo paragraph trước
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # Nếu một paragraph đơn lẻ đã vượt chunk_chars → cắt theo câu
        if para_len > chunk_chars:
            # Flush current buffer trước
            if current_chunk_parts:
                chunks.append({
                    "text": "\n\n".join(current_chunk_parts),
                    "metadata": {**base_metadata, "section": section},
                })
                # Overlap: giữ lại phần cuối của chunk trước
                overlap_text = current_chunk_parts[-1] if current_chunk_parts else ""
                current_chunk_parts = [overlap_text] if len(overlap_text) <= overlap_chars else []
                current_len = sum(len(p) for p in current_chunk_parts)

            # Cắt paragraph dài theo câu
            sentence_chunks = _split_long_paragraph(para, chunk_chars, overlap_chars)
            for sc in sentence_chunks:
                chunks.append({
                    "text": sc,
                    "metadata": {**base_metadata, "section": section},
                })
            continue

        # Nếu thêm paragraph này vượt quá chunk_chars → flush chunk hiện tại
        separator_len = 2 if current_chunk_parts else 0  # "\n\n"
        if current_len + separator_len + para_len > chunk_chars and current_chunk_parts:
            chunks.append({
                "text": "\n\n".join(current_chunk_parts),
                "metadata": {**base_metadata, "section": section},
            })
            # Overlap: giữ lại paragraph cuối của chunk trước
            last_part = current_chunk_parts[-1]
            if len(last_part) <= overlap_chars:
                current_chunk_parts = [last_part]
                current_len = len(last_part)
            else:
                current_chunk_parts = []
                current_len = 0

        current_chunk_parts.append(para)
        current_len += (separator_len + para_len)

    # Flush phần còn lại
    if current_chunk_parts:
        chunks.append({
            "text": "\n\n".join(current_chunk_parts),
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


def _split_long_paragraph(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> List[str]:
    """
    Fallback: cắt một paragraph quá dài theo ranh giới câu.
    """
    # Tách theo câu (dấu chấm, chấm hỏi, chấm than + khoảng trắng)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_parts: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        sep_len = 1 if current_parts else 0  # space

        if current_len + sep_len + sent_len > chunk_chars and current_parts:
            chunks.append(" ".join(current_parts))
            # Overlap: giữ câu cuối
            last = current_parts[-1]
            if len(last) <= overlap_chars:
                current_parts = [last]
                current_len = len(last)
            else:
                current_parts = []
                current_len = 0

        current_parts.append(sent)
        current_len += sep_len + sent_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================

# Cache model Sentence Transformers để không load lại mỗi lần
_st_model = None


def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.
    Dùng OpenAI nếu có API key, fallback về Sentence Transformers local.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        # Option A — OpenAI Embeddings
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    else:
        # Option B — Sentence Transformers (local, không cần API key)
        global _st_model
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return _st_model.encode(text).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store vào ChromaDB.
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo ChromaDB
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # Preprocess
        doc = preprocess_document(raw_text, str(filepath))

        # Chunk
        chunks = chunk_document(doc)
        print(f"    → {len(chunks)} chunks")

        # Embed và lưu từng chunk vào ChromaDB
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )

        total_chunks += len(chunks)

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")
    print(f"ChromaDB lưu tại: {db_dir}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Department: {meta.get('department', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Access: {meta.get('access', 'N/A')}")
            print(f"  Text preview: {doc[:150]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        total = len(results["metadatas"])
        print(f"\nTổng chunks: {total}")

        departments: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        missing_date = 0
        missing_section = 0

        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1

            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1
            if meta.get("section") in ("", None, "General"):
                missing_section += 1

        print("\nPhân bố theo department:")
        for dept, count in sorted(departments.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dept}: {count} chunks ({count*100//total}%)")

        print(f"\nPhân bố theo source:")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {src}: {count} chunks")

        print(f"\nChất lượng metadata:")
        print(f"  Chunks thiếu effective_date: {missing_date}/{total}")
        print(f"  Chunks thiếu section (General): {missing_section}/{total}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build full index
    print("\n--- Build Full Index ---")
    build_index()

    # Bước 4: Kiểm tra index
    list_chunks()
    inspect_metadata_coverage()

    print("\nSprint 1 hoàn thành!")